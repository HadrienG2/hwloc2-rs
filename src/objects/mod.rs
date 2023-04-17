//! Topology objects

// - Top-level doc: https://hwloc.readthedocs.io/en/v2.9/structhwloc__obj.html
// - Attributes: https://hwloc.readthedocs.io/en/v2.9/attributes.html

pub mod attributes;
pub mod depth;
pub mod distances;
pub mod types;

use self::{
    attributes::{DownstreamAttributes, ObjectAttributes, PCIDomain, RawObjectAttributes},
    depth::{Depth, DepthError, DepthResult, RawDepth},
    types::{CacheType, ObjectType, RawObjectType},
};
#[cfg(doc)]
use crate::topology::{builder::BuildFlags, support::DiscoverySupport};
use crate::{
    bitmaps::RawBitmap,
    cpu::sets::CpuSet,
    errors::NulError,
    ffi::{self, LibcString},
    info::TextualInfo,
    memory::nodesets::NodeSet,
    topology::Topology,
};
use num_enum::TryFromPrimitiveError;
use std::{
    ffi::{c_char, c_int, c_uint, c_void, CStr},
    fmt,
    iter::FusedIterator,
    ptr,
};

/// # Object levels, depths and types
///
/// Be sure to see read through the
/// [Terms and Definitions](https://hwloc.readthedocs.io/en/v2.9/termsanddefs.html)
/// section of the upstream hwloc documentation to avoid any confusion about
/// depths, child/sibling/cousin relationships, and see an example of an
/// asymmetric topology where one package has fewer caches than its peers.
//
// Upstream docs: https://hwloc.readthedocs.io/en/v2.9/group__hwlocality__levels.html
impl Topology {
    /// Depth of the hierarchical tree of objects
    ///
    /// This is the depth of [`ObjectType::PU`] plus one. NUMA nodes, I/O and
    /// Misc objects are ignored when computing the depth of the tree (they are
    /// placed on special levels).
    ///
    /// # Examples
    ///
    /// ```
    /// # let topology = hwlocality::Topology::test_instance();
    /// // The Machine and PU depths are always present
    /// assert!(topology.depth() >= 2);
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    #[doc(alias = "hwloc_topology_get_depth")]
    pub fn depth(&self) -> u32 {
        unsafe { ffi::hwloc_topology_get_depth(self.as_ptr()) }
            .try_into()
            .expect("Got unexpected depth from hwloc_topology_get_depth")
    }

    /// Depth of parents where memory objects are attached
    ///
    /// # Errors
    ///
    /// - [`DepthError::Multiple`] if memory objects are attached at multiple
    ///   depths
    ///
    /// # Examples
    ///
    /// ```
    /// # use hwlocality::objects::TopologyObject;
    /// # let topology = hwlocality::Topology::test_instance();
    /// if let Ok(depth) = topology.memory_parents_depth() {
    ///     let num_memory_objects =
    ///         topology.objects_at_depth(depth)
    ///                 .flat_map(TopologyObject::memory_children)
    ///                 .count();
    ///     assert!(num_memory_objects > 0);
    /// }
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    #[doc(alias = "hwloc_get_memory_parents_depth")]
    pub fn memory_parents_depth(&self) -> DepthResult {
        Depth::try_from(unsafe { ffi::hwloc_get_memory_parents_depth(self.as_ptr()) })
    }

    /// Depth for the given [`ObjectType`]
    ///
    /// # Errors
    ///
    /// - [`DepthError::None`] if no object of this type is present or
    ///   if the OS doesn't provide this kind of information. If a similar type
    ///   is acceptable, consider using [depth_or_below_for_type()] or
    ///   [depth_or_above_for_type()] instead.
    /// - [`DepthError::Multiple`] if objects of this type exist at multiple
    ///   depths.
    ///
    /// # Examples
    ///
    /// ```
    /// # use hwlocality::objects::types::ObjectType;
    /// #
    /// # let topology = hwlocality::Topology::test_instance();
    /// #
    /// let machine_depth = topology.depth_for_type(ObjectType::Machine)?;
    /// let pu_depth = topology.depth_for_type(ObjectType::PU)?;
    ///
    /// assert_eq!(machine_depth.assume_normal(), 0);
    /// assert!(machine_depth.assume_normal() < pu_depth.assume_normal());
    /// #
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    ///
    /// [depth_or_below_for_type()]: Topology::depth_or_below_for_type()
    /// [depth_or_above_for_type()]: Topology::depth_or_above_for_type()
    #[doc(alias = "hwloc_get_type_depth")]
    pub fn depth_for_type(&self, object_type: ObjectType) -> DepthResult {
        Depth::try_from(unsafe { ffi::hwloc_get_type_depth(self.as_ptr(), object_type.into()) })
    }

    /// Depth for the given [`ObjectType`] or below
    ///
    /// If no object of this type is present on the underlying architecture, the
    /// function returns the depth of the first present object typically found
    /// inside `object_type`.
    ///
    /// This function is only meaningful for normal object types.
    ///
    /// # Errors
    ///
    /// - [`DepthError::Multiple`] if objects of this type exist at multiple
    ///   depths
    ///
    /// # Examples
    ///
    /// ```
    /// # use hwlocality::{objects::types::ObjectType};
    /// #
    /// # let topology = hwlocality::Topology::test_instance();
    /// #
    /// let machine_depth = topology.depth_for_type(ObjectType::Machine)?;
    /// let package_or_below = topology.depth_or_below_for_type(ObjectType::Package)?;
    ///
    /// assert!(machine_depth.assume_normal() < package_or_below.assume_normal());
    /// #
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    #[doc(alias = "hwloc_get_type_or_below_depth")]
    pub fn depth_or_below_for_type(&self, object_type: ObjectType) -> DepthResult {
        assert!(
            object_type.is_normal(),
            "This is only meaningful for normal objects"
        );
        match self.depth_for_type(object_type) {
            Ok(d) => Ok(d),
            Err(DepthError::None) => {
                let pu_depth = self
                    .depth_for_type(ObjectType::PU)
                    .expect("PU objects should be present")
                    .assume_normal();
                for depth in (0..pu_depth).rev() {
                    if self
                        .type_at_depth(depth)
                        .expect("Depths above PU depth should exist")
                        < object_type
                    {
                        return Ok((depth + 1).into());
                    }
                }
                Err(DepthError::None)
            }
            other_err => other_err,
        }
    }

    /// Depth for the given [`ObjectType`] or above
    ///
    /// If no object of this type is present on the underlying architecture, the
    /// function returns the depth of the first present object typically
    /// containing `object_type`.
    ///
    /// This function is only meaningful for normal object types.
    ///
    /// # Errors
    ///
    /// - [`DepthError::Multiple`] if objects of this type exist at multiple
    ///   depths
    ///
    /// # Examples
    ///
    /// ```
    /// # use hwlocality::objects::types::ObjectType;
    /// #
    /// # let topology = hwlocality::Topology::test_instance();
    /// #
    /// let pu_depth = topology.depth_for_type(ObjectType::PU)?;
    /// let core_or_above = topology.depth_or_below_for_type(ObjectType::Core)?;
    ///
    /// assert!(core_or_above.assume_normal() < pu_depth.assume_normal());
    /// #
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    #[doc(alias = "hwloc_get_type_or_above_depth")]
    pub fn depth_or_above_for_type(&self, object_type: ObjectType) -> DepthResult {
        assert!(
            object_type.is_normal(),
            "This is only meaningful for normal objects"
        );
        match self.depth_for_type(object_type) {
            Ok(d) => Ok(d),
            Err(DepthError::None) => {
                for depth in (0..self.depth()).rev() {
                    if self
                        .type_at_depth(depth)
                        .expect("Depths above bottom depth should exist")
                        > object_type
                    {
                        return Ok((depth - 1).into());
                    }
                }
                Err(DepthError::None)
            }
            other_err => other_err,
        }
    }

    /// Depth for the given cache type and level
    ///
    /// Return the depth of the topology level that contains cache objects whose
    /// attributes match `cache_level` and `cache_type`.
    ///
    /// This function is similar to calling [depth_for_type()] with
    /// the corresponding type such as [`ObjectType::L1ICache`], except that it
    /// may also return a unified cache when looking for an instruction cache.
    ///
    /// If `cache_type` is `None`, it is ignored and multiple levels may match.
    /// The function returns either the depth of a uniquely matching level or
    /// Err([`DepthError::Multiple`]).
    ///
    /// If `cache_type` is Some([`CacheType::Unified`]), the depth of the unique
    /// matching unified cache level (if any) is returned.
    ///
    /// If `cache_type` is Some([`CacheType::Data`]) or
    /// Some([`CacheType::Instruction`]), either a matching cache or a
    /// unified cache is returned.
    ///
    /// # Errors
    ///
    /// - [`DepthError::None`] if no cache level matches
    /// - [`DepthError::Multiple`] if multiple cache depths match (this can only
    ///   happen if `cache_type` is `None`).
    ///
    /// # Examples
    ///
    /// ```
    /// # use hwlocality::objects::types::CacheType;
    /// # let topology = hwlocality::Topology::test_instance();
    /// let l1d_depth = topology.depth_for_cache(1, Some(CacheType::Data));
    /// assert!(l1d_depth.is_ok());
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    ///
    /// [depth_for_type()]: Topology::depth_for_type()
    #[doc(alias = "hwloc_get_cache_type_depth")]
    pub fn depth_for_cache(&self, cache_level: u32, cache_type: Option<CacheType>) -> DepthResult {
        let mut result = Err(DepthError::None);
        for depth in 0..self.depth() {
            // Cache level and type are homogeneous across a depth level so we
            // only need to look at one object
            for obj in self.objects_at_depth(depth).take(1) {
                // Is this a cache?
                if let Some(ObjectAttributes::Cache(cache)) = obj.attributes() {
                    // Check cache level
                    if cache.depth() != cache_level {
                        continue;
                    }

                    // Check cache type if instructed to do so
                    if let Some(cache_type) = cache_type {
                        if cache.cache_type() == cache_type
                            || cache.cache_type() == CacheType::Unified
                        {
                            // If both cache type + level are specified, then
                            // multiple matches cannot occur: stop here.
                            return Ok(depth.into());
                        } else {
                            continue;
                        }
                    } else {
                        // Without a cache type check, multiple matches may
                        // occur, so we need to check all other depths.
                        match result {
                            Err(DepthError::None) => result = Ok(depth.into()),
                            Ok(_) => {
                                return Err(DepthError::Multiple);
                            }
                            Err(DepthError::Multiple) => {
                                unreachable!("Setting this value triggers a loop break")
                            }
                            Err(DepthError::Unknown(_)) => {
                                unreachable!("This value is never set")
                            }
                        }
                    }
                }
            }
        }
        result
    }

    /// [`ObjectType`] at the given `depth`
    ///
    /// # Examples
    ///
    /// ```
    /// # use hwlocality::objects::{depth::Depth, types::ObjectType};
    /// # let topology = hwlocality::Topology::test_instance();
    /// let numa_type = topology.type_at_depth(Depth::NUMANode);
    /// assert_eq!(numa_type, Some(ObjectType::NUMANode));
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    #[doc(alias = "hwloc_get_depth_type")]
    pub fn type_at_depth(&self, depth: impl Into<Depth>) -> Option<ObjectType> {
        let depth = depth.into();
        if let Depth::Normal(depth) = depth {
            if depth >= self.depth() {
                return None;
            }
        }
        match unsafe { ffi::hwloc_get_depth_type(self.as_ptr(), depth.into()) }.try_into() {
            Ok(depth) => Some(depth),
            Err(TryFromPrimitiveError {
                number: RawObjectType::MAX,
            }) => None,
            Err(unknown) => {
                unreachable!("Got unknown object type from hwloc_get_depth_type: {unknown}")
            }
        }
    }

    /// Number of objects at the given `depth`
    ///
    /// # Examples
    ///
    /// ```
    /// # let topology = hwlocality::Topology::test_instance();
    /// #
    /// let num_roots = topology.size_at_depth(0);
    /// assert_eq!(num_roots, 1);
    ///
    /// let num_root_children = topology.size_at_depth(1);
    /// assert!(num_root_children > 0);
    /// #
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    #[doc(alias = "hwloc_get_nbobjs_by_depth")]
    pub fn size_at_depth(&self, depth: impl Into<Depth>) -> u32 {
        unsafe { ffi::hwloc_get_nbobjs_by_depth(self.as_ptr(), depth.into().into()) }
    }

    /// [`TopologyObject`]s at the given `depth`
    ///
    /// # Examples
    ///
    /// ```
    /// # use hwlocality::objects::{depth::Depth, types::ObjectType};
    /// # let topology = hwlocality::Topology::test_instance();
    /// #
    /// use anyhow::Context;
    ///
    /// let root = topology.root_object();
    ///
    /// for node in topology.objects_at_depth(Depth::NUMANode) {
    ///     assert_eq!(node.object_type(), ObjectType::NUMANode);
    ///     assert!(node.is_in_subtree(root));
    ///     assert_eq!(node.normal_arity(), 0);
    ///     assert_eq!(node.memory_arity(), 0);
    ///     let num_nodes =
    ///         node.nodeset().context("A NUMANode should have a NodeSet")?
    ///             .weight().context("A NUMANode's NodeSet should be finite")?;
    ///     assert_eq!(num_nodes, 1);
    /// }
    /// #
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    #[doc(alias = "hwloc_get_obj_by_depth")]
    #[doc(alias = "hwloc_get_next_obj_by_depth")]
    pub fn objects_at_depth(
        &self,
        depth: impl Into<Depth>,
    ) -> impl Iterator<Item = &TopologyObject>
           + Clone
           + DoubleEndedIterator
           + ExactSizeIterator
           + FusedIterator {
        let depth = depth.into();
        let size = self.size_at_depth(depth);
        let depth = RawDepth::from(depth);
        (0..size).map(move |idx| {
            let ptr = unsafe { ffi::hwloc_get_obj_by_depth(self.as_ptr(), depth, idx) };
            assert!(
                !ptr.is_null(),
                "Got null pointer from hwloc_get_obj_by_depth"
            );
            unsafe { &*ptr }
        })
    }

    /// [`TopologyObject`] at the root of the topology
    ///
    /// # Examples
    ///
    /// ```
    /// # use hwlocality::objects::{depth::Depth, types::ObjectType};
    /// # let topology = hwlocality::Topology::test_instance();
    /// let root = topology.root_object();
    ///
    /// assert_eq!(root.object_type(), ObjectType::Machine);
    ///
    /// assert_eq!(root.depth(), Depth::Normal(0));
    /// assert!(root.parent().is_none());
    /// assert_eq!(root.logical_index(), 0);
    /// assert_ne!(root.normal_arity(), 0);
    ///
    /// assert!(root.cpuset().is_some());
    /// assert!(root.nodeset().is_some());
    ///
    /// println!("{root:#}");
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    #[doc(alias = "hwloc_get_root_obj")]
    pub fn root_object(&self) -> &TopologyObject {
        self.objects_at_depth(0)
            .next()
            .expect("Root object should exist")
    }

    /// [`TopologyObject`]s with the given [`ObjectType`]
    ///
    /// # Examples
    ///
    /// ```
    /// # use hwlocality::objects::types::ObjectType;
    /// # let topology = hwlocality::Topology::test_instance();
    /// #
    /// use anyhow::Context;
    ///
    /// let root = topology.root_object();
    ///
    /// for pu in topology.objects_with_type(ObjectType::PU) {
    ///     assert_eq!(pu.object_type(), ObjectType::PU);
    ///     assert!(pu.is_in_subtree(root));
    ///     assert_eq!(pu.normal_arity(), 0);
    ///     let num_cpus =
    ///         pu.cpuset().context("A PU should have a CpuSet")?
    ///           .weight().context("A PU's CpuSet should be finite")?;
    ///     assert_eq!(num_cpus, 1);
    /// }
    /// #
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    #[doc(alias = "hwloc_get_obj_by_type")]
    #[doc(alias = "hwloc_get_next_obj_by_type")]
    pub fn objects_with_type(
        &self,
        object_type: ObjectType,
    ) -> impl Iterator<Item = &TopologyObject>
           + Clone
           + DoubleEndedIterator
           + ExactSizeIterator
           + FusedIterator {
        let type_depth = self.depth_for_type(object_type);
        let depth_iter = (0..self.depth())
            .map(Depth::from)
            .chain(Depth::VIRTUAL_DEPTHS.iter().copied())
            .filter(move |&depth| {
                if let Ok(type_depth) = type_depth {
                    depth == type_depth
                } else {
                    self.type_at_depth(depth).expect("Depth should exist") == object_type
                }
            });
        let size = depth_iter
            .clone()
            .map(move |depth| {
                usize::try_from(self.size_at_depth(depth)).expect("Impossible object count")
            })
            .sum();
        ObjectsWithType {
            size,
            inner: depth_iter.flat_map(move |depth| self.objects_at_depth(depth)),
        }
    }
}

/// Iterator emitted by objects_with_type
#[derive(Copy, Clone)]
struct ObjectsWithType<Inner> {
    size: usize,
    inner: Inner,
}
//
impl<'topology, Inner: Iterator<Item = &'topology TopologyObject>> Iterator
    for ObjectsWithType<Inner>
{
    type Item = &'topology TopologyObject;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.size, Some(self.size))
    }

    fn count(self) -> usize
    where
        Self: Sized,
    {
        self.size
    }
}
//
impl<'topology, Inner: Iterator<Item = &'topology TopologyObject> + DoubleEndedIterator>
    DoubleEndedIterator for ObjectsWithType<Inner>
{
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.next_back()
    }
}
//
impl<'topology, Inner: Iterator<Item = &'topology TopologyObject>> ExactSizeIterator
    for ObjectsWithType<Inner>
{
}
//
impl<'topology, Inner: Iterator<Item = &'topology TopologyObject> + FusedIterator> FusedIterator
    for ObjectsWithType<Inner>
{
}

/// # Finding other objects
//
// This is inspired by the upstream functionality described at
// https://hwloc.readthedocs.io/en/v2.9/group__hwlocality__helper__find__misc.html
// but the code had to be ported to Rust because it's inline
impl Topology {
    /// Get the object of type [`ObjectType::PU`] with the specified OS index
    ///
    /// If you want to convert an entire CPU set into the PU objects it
    /// contains, using `pus_from_cpuset` will be more efficient than repeatedly
    /// calling this function with every OS index from the CpuSet.
    ///
    /// Requires [`DiscoverySupport::pu_count()`].
    pub fn pu_with_os_index(&self, os_index: u32) -> Option<&TopologyObject> {
        self.objs_and_os_indices(ObjectType::PU)
            .find_map(|(pu, pu_os_index)| (pu_os_index == os_index).then_some(pu))
    }

    /// Get the objects of type [`ObjectType::PU`] covered by the specified cpuset
    ///
    /// Requires [`DiscoverySupport::pu_count()`].
    pub fn pus_from_cpuset<'result>(
        &'result self,
        cpuset: &'result CpuSet,
    ) -> impl Iterator<Item = &TopologyObject> + Clone + DoubleEndedIterator + FusedIterator + 'result
    {
        self.objs_and_os_indices(ObjectType::PU)
            .filter_map(|(pu, os_index)| cpuset.is_set(os_index).then_some(pu))
    }

    /// Get the object of type [`ObjectType::NUMANode`] with the specified OS index
    ///
    /// If you want to convert an entire NodeSet into the NUMANode objects it
    /// contains, using `nodes_from_cpuset` will be more efficient than repeatedly
    /// calling this function with every OS index from the NodeSet.
    ///
    /// Requires [`DiscoverySupport::numa_count()`].
    pub fn node_with_os_index(&self, os_index: u32) -> Option<&TopologyObject> {
        self.objs_and_os_indices(ObjectType::NUMANode)
            .find_map(|(node, node_os_index)| (node_os_index == os_index).then_some(node))
    }

    /// Get the objects of type [`ObjectType::NUMANode`] covered by the
    /// specified nodeset
    ///
    /// Requires [`DiscoverySupport::numa_count()`].
    pub fn nodes_from_nodeset<'result>(
        &'result self,
        nodeset: &'result NodeSet,
    ) -> impl Iterator<Item = &TopologyObject> + Clone + DoubleEndedIterator + FusedIterator + 'result
    {
        self.objs_and_os_indices(ObjectType::NUMANode)
            .filter_map(|(node, os_index)| nodeset.is_set(os_index).then_some(node))
    }

    /// Get a list of `(&TopologyObject, OS index)` tuples for an `ObjectType`
    /// that is guaranteed to appear only at one depth of the topology and to
    /// have an OS index.
    ///
    /// # Panics
    ///
    /// Will panic if the object type appears at more than one depth or do not
    /// have an OS index.
    fn objs_and_os_indices(
        &self,
        ty: ObjectType,
    ) -> impl Iterator<Item = (&TopologyObject, u32)>
           + Clone
           + DoubleEndedIterator
           + ExactSizeIterator
           + FusedIterator {
        self.objects_at_depth(
            self.depth_for_type(ty)
                .expect("These objects should only appear at a single depth"),
        )
        .map(|obj| {
            (
                obj,
                obj.os_index()
                    .expect("These objects should have an OS index"),
            )
        })
    }

    /// Enumerate objects at the same depth as `obj`, but with increasing
    /// physical distance (i.e. from increasingly higher common ancestors in the
    /// topology tree)
    ///
    /// # Panics
    ///
    /// `obj` must have a cpuset, otherwise this function will panic.
    pub fn closest_objects<'result>(
        &'result self,
        obj: &'result TopologyObject,
    ) -> impl Iterator<Item = &TopologyObject> + Clone + 'result {
        // Track which CPUs map into objects we don't want to report
        // (current object or already reported object)
        let mut known_cpuset = obj.cpuset().expect("Target object must have a cpuset");

        // Assert that an object has a cpuset, return both
        fn obj_and_cpuset<'obj>(
            obj: &'obj TopologyObject,
            error: &str,
        ) -> (&'obj TopologyObject, &'obj CpuSet) {
            (obj, obj.cpuset().expect(error))
        }

        // Find the first ancestor of an object that knows about more objects
        // than that object (if any), and return it along with its cpuset
        fn find_larger_parent<'obj>(
            known_obj: &'obj TopologyObject,
            known_cpuset: &CpuSet,
        ) -> Option<(&'obj TopologyObject, &'obj CpuSet)> {
            known_obj
                .ancestors()
                .map(|ancestor| {
                    obj_and_cpuset(
                        ancestor,
                        "Ancestors of an obj with a cpuset should have a cpuset",
                    )
                })
                .find(|&(_ancestor, ancestor_cpuset)| ancestor_cpuset != known_cpuset)
        }
        let mut ancestor_and_cpuset = find_larger_parent(obj, known_cpuset);

        // Prepare to jointly iterate over cousins and their cpusets
        let cousins_and_cpusets = self
            .objects_at_depth(obj.depth())
            .map(|cousin| {
                obj_and_cpuset(
                    cousin,
                    "Cousins of an obj with a cpuset should have a cpuset",
                )
            })
            .collect::<Vec<_>>();
        let mut cousin_idx = 0;

        // Emit the final iterator
        std::iter::from_fn(move || {
            loop {
                // Look for a cousin that is part of ancestor_cpuset but not known_cpuset
                let (ancestor, ancestor_cpuset) = ancestor_and_cpuset?;
                while let Some((cousin, cousin_cpuset)) = cousins_and_cpusets.get(cousin_idx) {
                    cousin_idx += 1;
                    if ancestor_cpuset.includes(cousin_cpuset)
                        && !known_cpuset.includes(cousin_cpuset)
                    {
                        return Some(*cousin);
                    }
                }

                // We ran out of cousins, go up one ancestor level or end
                // iteration if we reached the top of the tree.
                let known_obj = ancestor;
                known_cpuset = ancestor_cpuset;
                let (ancestor, ancestor_cpuset) = find_larger_parent(known_obj, known_cpuset)?;
                ancestor_and_cpuset = Some((ancestor, ancestor_cpuset));
                cousin_idx = 0;
            }
        })
    }

    /// Find an object via a parent->child chain specified by types and indices
    ///
    /// For example, if called with `&[(NUMANode, 0), (Package, 1), (Core, 2)]`,
    /// this will return the third core object below the second package below
    /// the first NUMA node.
    ///
    /// # Panics
    ///
    /// All objects must have a cpuset, otherwise this function will panic.
    pub fn object_by_type_index_path(
        &self,
        path: &[(ObjectType, usize)],
    ) -> Option<&TopologyObject> {
        let mut obj = self.root_object();
        for &(ty, idx) in path {
            let cpuset = obj
                .cpuset()
                .expect("All objects in path should have a cpuset");

            obj = self.objects_inside_cpuset_with_type(cpuset, ty).nth(idx)?;
        }
        Some(obj)
    }

    /// Find an object of a different type with the same locality
    ///
    /// If the source object src is a normal or memory type, this function
    /// returns an object of type type with same CPU and node sets, either below
    /// or above in the hierarchy.
    ///
    /// If the source object src is a PCI or an OS device within a PCI device,
    /// the function may either return that PCI device, or another OS device in
    /// the same PCI parent. This may for instance be useful for converting
    /// between OS devices such as "nvml0" or "rsmi1" used in distance
    /// structures into the the PCI device, or the CUDA or OpenCL OS device that
    /// correspond to the same physical card.
    ///
    /// If specified, parameter `subtype` restricts the search to objects whose
    /// [`TopologyObject::subtype()`] attribute exists and is equal to `subtype`
    /// (case-insensitively), for instance "OpenCL" or "CUDA".
    ///
    /// If specified, parameter `name_prefix` restricts the search to objects
    /// whose [`TopologyObject::name()`] attribute exists and starts with
    /// `name_prefix` (case-insensitively), for instance "rsmi" for matching
    /// "rsmi0".
    ///
    /// If multiple objects match, the first one is returned.
    ///
    /// This function will not walk the hierarchy across bridges since the PCI
    /// locality may become different. This function cannot also convert between
    /// normal/memory objects and I/O or Misc objects.
    ///
    /// If no matching object could be found, or if the source object and target
    /// type are incompatible, `None` will be returned.
    ///
    /// # Errors
    ///
    /// - [`NulError`] if `subtype` or `name_prefix` contains NUL chars.
    #[cfg(feature = "hwloc-2_5_0")]
    #[doc(alias = "hwloc_get_obj_with_same_locality")]
    pub fn object_with_same_locality(
        &self,
        src: &TopologyObject,
        ty: ObjectType,
        subtype: Option<&str>,
        name_prefix: Option<&str>,
    ) -> Result<Option<&TopologyObject>, NulError> {
        let subtype = subtype.map(LibcString::new).transpose()?;
        let name_prefix = name_prefix.map(LibcString::new).transpose()?;
        let borrow_pchar = |opt: &Option<LibcString>| -> *const c_char {
            opt.as_ref().map(|s| s.borrow()).unwrap_or(ptr::null())
        };
        let ptr = unsafe {
            ffi::hwloc_get_obj_with_same_locality(
                self.as_ptr(),
                src,
                ty.into(),
                borrow_pchar(&subtype),
                borrow_pchar(&name_prefix),
                0,
            )
        };
        Ok((!ptr.is_null()).then(|| unsafe { &*ptr }))
    }
}

/// # Finding I/O objects
//
// Inspired by https://hwloc.readthedocs.io/en/v2.9/group__hwlocality__advanced__io.html
// but inline functions had to be reimplemented in Rust
impl Topology {
    /// Enumerate PCI devices in the system
    #[doc(alias = "hwloc_get_next_pcidev")]
    pub fn pci_devices(
        &self,
    ) -> impl Iterator<Item = &TopologyObject>
           + Clone
           + DoubleEndedIterator
           + ExactSizeIterator
           + FusedIterator {
        self.objects_at_depth(Depth::PCIDevice)
    }

    /// Find the PCI device object matching the PCI bus id given domain, bus
    /// device and function PCI bus id
    #[doc(alias = "hwloc_get_pcidev_by_busid")]
    pub fn pci_device_by_bus_id(
        &self,
        domain: PCIDomain,
        bus_id: u8,
        bus_device: u8,
        function: u8,
    ) -> Option<&TopologyObject> {
        self.pci_devices().find(|obj| {
            let Some(ObjectAttributes::PCIDevice(pci)) = obj.attributes() else { unreachable!("All PCI devices should have PCI attributes") };
            pci.domain() == domain && pci.bus_id() == bus_id && pci.bus_device() == bus_device && pci.function() == function
        })
    }

    /// Find the PCI device object matching the PCI bus id given as a string
    /// of format "xxxx:yy:zz.t" (with domain) or "yy:zz.t" (without domain).
    ///
    /// # Panics
    ///
    /// If the given string does not match the PCI bus id format given above
    #[doc(alias = "hwloc_get_pcidev_by_busidstring")]
    pub fn pci_device_by_bus_id_string(&self, bus_id: &str) -> Option<&TopologyObject> {
        // Assume well-formatted string
        let parse_domain = |s| PCIDomain::from_str_radix(s, 16).expect("Bad hex domain format");
        let parse_u8 = |s| u8::from_str_radix(s, 16).expect("Bad hex u8 format");

        // Extract initial hex (whose semantics are ambiguous at this stage)
        let (int1, mut rest) = bus_id.split_once(':').expect("Bad address structure");

        // From presence/absence of second ':', deduce if int1 was a domain or
        // a bus id in the default 0 domain.
        let (domain, bus) = if let Some((bus, next_rest)) = rest.split_once(':') {
            rest = next_rest;
            (parse_domain(int1), parse_u8(bus))
        } else {
            (0, parse_u8(int1))
        };

        // Parse device and function IDs, and forward to non-textual lookup
        let (dev, func) = rest.split_once('.').expect("Bad address structure");
        self.pci_device_by_bus_id(domain, bus, parse_u8(dev), parse_u8(func))
    }

    /// Enumerate OS devices in the system
    #[doc(alias = "hwloc_get_next_osdev")]
    pub fn os_devices(
        &self,
    ) -> impl Iterator<Item = &TopologyObject>
           + Clone
           + DoubleEndedIterator
           + ExactSizeIterator
           + FusedIterator {
        self.objects_at_depth(Depth::OSDevice)
    }

    /// Enumerate bridges in the system
    #[doc(alias = "hwloc_get_next_bridge")]
    pub fn bridges(
        &self,
    ) -> impl Iterator<Item = &TopologyObject>
           + Clone
           + DoubleEndedIterator
           + ExactSizeIterator
           + FusedIterator {
        self.objects_at_depth(Depth::Bridge)
    }
}

/// Hardware topology object
///
/// Like `Topology`, this is a pretty big struct, so the documentation is
/// sliced into smaller parts:
///
/// - [Basic identity](#basic-identity)
/// - [Depth and ancestors](#depth-and-ancestors)
/// - [Cousins and siblings](#cousins-and-siblings)
/// - [Children](#children)
/// - [CPU set](#cpu-set)
/// - [NUMA node set](#numa-node-set)
/// - [Key-value information](#key-value-information)
#[doc(alias = "hwloc_obj")]
#[doc(alias = "hwloc_obj_t")]
#[repr(C)]
pub struct TopologyObject {
    // See the matching method names for more details on field semantics
    object_type: RawObjectType,
    subtype: *mut c_char,
    os_index: c_uint,
    name: *mut c_char,
    total_memory: u64,
    attr: *mut RawObjectAttributes,
    depth: RawDepth,
    logical_index: c_uint,
    next_cousin: *mut TopologyObject,
    prev_cousin: *mut TopologyObject,
    parent: *mut TopologyObject,
    sibling_rank: c_uint,
    next_sibling: *mut TopologyObject,
    prev_sibling: *mut TopologyObject,
    arity: c_uint,
    children: *mut *mut TopologyObject,
    first_child: *mut TopologyObject,
    last_child: *mut TopologyObject,
    symmetric_subtree: c_int,
    memory_arity: c_uint,
    memory_first_child: *mut TopologyObject,
    io_arity: c_uint,
    io_first_child: *mut TopologyObject,
    misc_arity: c_uint,
    misc_first_child: *mut TopologyObject,
    cpuset: *mut RawBitmap,
    complete_cpuset: *mut RawBitmap,
    nodeset: *mut RawBitmap,
    complete_nodeset: *mut RawBitmap,
    infos: *mut TextualInfo,
    infos_count: c_uint,
    __userdata: *mut c_void, // BEWARE: Topology duplication blindly duplicates this!
    gp_index: u64,
}

/// # Basic identity
impl TopologyObject {
    /// Type of object.
    pub fn object_type(&self) -> ObjectType {
        self.object_type
            .try_into()
            .expect("Got unexpected object type")
    }

    /// Subtype string to better describe the type field
    ///
    /// See <https://hwloc.readthedocs.io/en/v2.9/attributes.html#attributes_normal>
    /// for a list of subtype strings that hwloc can emit.
    pub fn subtype(&self) -> Option<&CStr> {
        unsafe { ffi::deref_str(&self.subtype) }
    }

    /// Set the subtype string
    ///
    /// This is something you'll often want to do when creating Group or Misc
    /// objects in order to make them more descriptive.
    ///
    /// # Errors
    ///
    /// - [`NulError`] if `subtype` contains NUL chars.
    pub fn set_subtype(&mut self, subtype: &str) -> Result<(), NulError> {
        self.subtype = LibcString::new(subtype)?.into_raw();
        Ok(())
    }

    /// The OS-provided physical index number.
    ///
    /// It is not guaranteed unique across the entire machine,
    /// except for PUs and NUMA nodes.
    ///
    /// Not specified if unknown or irrelevant for this object.
    pub fn os_index(&self) -> Option<u32> {
        const HWLOC_UNKNOWN_INDEX: c_uint = c_uint::MAX;
        (self.os_index != HWLOC_UNKNOWN_INDEX).then_some(self.os_index)
    }

    /// The name of the object
    pub fn name(&self) -> Option<&CStr> {
        unsafe { ffi::deref_str(&self.name) }
    }

    /// Object type-specific attributes
    pub fn attributes(&self) -> Option<ObjectAttributes> {
        unsafe { ObjectAttributes::new(self.object_type(), &self.attr) }
    }

    /// Unsafe access to object type-specific attributes
    #[cfg(feature = "hwloc-2_3_0")]
    pub(crate) fn raw_attributes(&mut self) -> Option<&mut RawObjectAttributes> {
        unsafe { ffi::deref_mut_ptr(&mut self.attr) }
    }
}

/// # Depth and ancestors
impl TopologyObject {
    /// Vertical index in the hierarchy
    ///
    /// For normal objects, this is the depth of the horizontal level that
    /// contains this object and its cousins of the same type. If the topology
    /// is symmetric, this is equal to the parent depth plus one, and also equal
    /// to the number of parent/child links from the root object to here.
    ///
    /// For special objects (NUMA nodes, I/O and Misc) that are not in the main
    /// tree, this is a special value that is unique to their type.
    pub fn depth(&self) -> Depth {
        self.depth.try_into().expect("Got unexpected depth value")
    }

    /// Parent object
    pub fn parent(&self) -> Option<&TopologyObject> {
        unsafe { ffi::deref_ptr_mut(&self.parent) }
    }

    /// Chain of parent objects up to the topology root
    pub fn ancestors(
        &self,
    ) -> impl Iterator<Item = &TopologyObject> + Copy + ExactSizeIterator + FusedIterator {
        Ancestors(self)
    }

    /// Search for an ancestor at a certain depth
    ///
    /// Will return `None` if the requested depth is deeper than the depth of
    /// the current object.
    pub fn ancestor_at_depth(&self, depth: impl Into<Depth>) -> Option<&TopologyObject> {
        // Fast failure path when depth is comparable
        let depth = depth.into();
        let self_depth = self.depth();
        if let (Ok(self_depth), Ok(depth)) = (u32::try_from(self_depth), u32::try_from(depth)) {
            if self_depth <= depth {
                return None;
            }
        }

        // Otherwise, walk parents looking for the right depth
        self.ancestors().find(|ancestor| ancestor.depth() == depth)
    }

    /// Search for the first ancestor with a certain type in ascending order
    ///
    /// Will return `None` if the requested type appears deeper than the
    /// current object (e.g. `PU`) or doesn't appear in the topology.
    pub fn first_ancestor_with_type(&self, ty: ObjectType) -> Option<&TopologyObject> {
        self.ancestors()
            .find(|ancestor| ancestor.object_type() == ty)
    }

    /// Search for an ancestor that is shared with another object
    ///
    /// # Panics
    ///
    /// If one of the objects has a special depth (memory, I/O...).
    // FIXME: It should actually be possible to handle that without panicking
    //        by collecting the list of ancestors up to the first ancestor that
    //        has a normal depth, then looking up common patterns, and if that
    //        fails resuming the normal algorithm.
    pub fn common_ancestor(&self, other: &TopologyObject) -> Option<&TopologyObject> {
        // Handle degenerate case
        if ptr::eq(self, other) {
            return self.parent();
        }

        // Otherwise, follow hwloc's example, but restrict it to normal depths
        // as I don't think their algorithm is correct for special depths.
        let u32_depth = |obj: &TopologyObject| {
            u32::try_from(obj.depth()).expect("Need normal depth for this algorithm")
        };
        let mut parent1 = self.parent()?;
        let mut parent2 = other.parent()?;
        loop {
            // Walk up parent1 and parent2 ancestors, try to reach the same depth
            let depth2 = u32_depth(parent2);
            while u32_depth(parent1) > depth2 {
                parent1 = parent1.parent()?;
            }
            let depth1 = u32_depth(parent1);
            while u32_depth(parent2) > depth1 {
                parent2 = parent2.parent()?;
            }

            // If we reached the same parent, we're done
            if ptr::eq(parent1, parent2) {
                return Some(parent1);
            }

            // Otherwise, either parent2 jumped above parent1 (which can happen
            // as hwloc topology may "skip" depths on hybrid plaforms like
            // Adler Lake or in the presence of complicated allowed cpusets), or
            // we reached cousin objects and must go up one level.
            if parent1.depth == parent2.depth {
                parent1 = parent1.parent()?;
                parent2 = parent2.parent()?;
            }
        }
    }

    /// Truth that this object is in the subtree beginning with ancestor
    /// object `subtree_root`
    pub fn is_in_subtree(&self, subtree_root: &TopologyObject) -> bool {
        // Take a cpuset-based shortcut on normal objects
        if let (Some(self_cpuset), Some(subtree_cpuset)) = (self.cpuset(), subtree_root.cpuset()) {
            return subtree_cpuset.includes(self_cpuset);
        }

        // Otherwise, walk the ancestor chain
        self.ancestors()
            .any(|ancestor| ptr::eq(ancestor, subtree_root))
    }

    /// Get the first data (or unified) CPU cache shared between this object and
    /// another object, if any.
    ///
    /// Will always return `None` if called on an I/O or Misc object that does
    /// not contain CPUs.
    pub fn first_shared_cache(&self) -> Option<&TopologyObject> {
        let cpuset = self.cpuset()?;
        self.ancestors()
            .skip_while(|ancestor| ancestor.cpuset() == Some(cpuset))
            .find(|ancestor| ancestor.object_type().is_cpu_data_cache())
    }

    /// Get the first non-I/O ancestor object
    ///
    /// Find the smallest non-I/O ancestor object. This object (normal or
    /// memory) may then be used for binding because it has CPU and node sets
    /// and because its locality is the same as this object
    #[doc(alias = "hwloc_get_non_io_ancestor_obj")]
    pub fn non_io_ancestor(&self) -> Option<&TopologyObject> {
        self.ancestors().find(|obj| obj.cpuset().is_some())
    }
}

/// Iterator over ancestors of a TopologyObject
#[derive(Copy, Clone, Debug)]
struct Ancestors<'object>(&'object TopologyObject);
//
impl<'object> Iterator for Ancestors<'object> {
    type Item = &'object TopologyObject;

    fn next(&mut self) -> Option<Self::Item> {
        self.0 = self.0.parent()?;
        Some(self.0)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let depth_res = u32::try_from(self.0.depth())
            .map(|depth| usize::try_from(depth).expect("Depth deeper than 2^32 not unexpected"));
        (depth_res.unwrap_or(0), depth_res.ok())
    }
}
//
impl ExactSizeIterator for Ancestors<'_> {}
//
impl FusedIterator for Ancestors<'_> {}

/// # Cousins and siblings
impl TopologyObject {
    /// Horizontal index in the whole list of similar objects, hence guaranteed
    /// unique across the entire machine.
    ///
    /// Could be a "cousin_rank" since it's the rank within the "cousin" list below.
    ///
    /// Note that this index may change when restricting the topology
    /// or when inserting a group.
    pub fn logical_index(&self) -> u32 {
        self.logical_index
    }

    /// Next object of same type and depth
    pub fn next_cousin(&self) -> Option<&TopologyObject> {
        unsafe { ffi::deref_ptr_mut(&self.next_cousin) }
    }

    /// Previous object of same type and depth
    pub fn prev_cousin(&self) -> Option<&TopologyObject> {
        unsafe { ffi::deref_ptr_mut(&self.prev_cousin) }
    }

    /// Index in the parent's appropriate child list
    pub fn sibling_rank(&self) -> u32 {
        self.sibling_rank
    }

    /// Next object below the same parent in the same child list
    pub fn next_sibling(&self) -> Option<&TopologyObject> {
        unsafe { ffi::deref_ptr_mut(&self.next_sibling) }
    }

    /// Previous object below the same parent in the same child list
    pub fn prev_sibling(&self) -> Option<&TopologyObject> {
        unsafe { ffi::deref_ptr_mut(&self.prev_sibling) }
    }
}

/// # Children
impl TopologyObject {
    /// Number of normal children (excluding Memory, Misc and I/O)
    pub fn normal_arity(&self) -> u32 {
        self.arity
    }

    /// Normal children of this object
    pub fn normal_children(
        &self,
    ) -> impl Iterator<Item = &TopologyObject>
           + Clone
           + DoubleEndedIterator
           + ExactSizeIterator
           + FusedIterator {
        if self.children.is_null() {
            assert_eq!(
                self.normal_arity(),
                0,
                "Got null children pointer with nonzero arity"
            );
        }
        (0..self.normal_arity()).map(move |i| {
            // If this fails, it means self.arity does not fit in a
            // size_t, but by definition of size_t that cannot happen...
            let offset = isize::try_from(i).expect("Should not happen");
            let child = unsafe { *self.children.offset(offset) };
            assert!(!child.is_null(), "Got null child pointer");
            unsafe { &*child }
        })
    }

    /// Truth that this object is symmetric, which means all normal children and
    /// their children have identical subtrees.
    ///
    /// Memory, I/O and Misc children are ignored.
    pub fn symmetric_subtree(&self) -> bool {
        self.symmetric_subtree != 0
    }

    /// Get the child covering at least the given cpuset `set`
    ///
    /// This function will always return `None` if the given set is empty or
    /// this TopologyObject doesn't have a cpuset (I/O or Misc objects).
    pub fn normal_child_covering_cpuset(&self, set: &CpuSet) -> Option<&TopologyObject> {
        self.normal_children()
            .find(|child| child.covers_cpuset(set))
    }

    /// Number of memory children
    pub fn memory_arity(&self) -> u32 {
        dbg!(self.memory_arity)
    }

    /// Memory children of this object
    ///
    /// NUMA nodes and Memory-side caches are listed here instead of in the
    /// [`TopologyObject::normal_children()`] list. See also
    /// [`ObjectType::is_memory()`].
    ///
    /// A memory hierarchy starts from a normal CPU-side object (e.g. Package)
    /// and ends with NUMA nodes as leaves. There might exist some memory-side
    /// caches between them in the middle of the memory subtree.
    pub fn memory_children(
        &self,
    ) -> impl Iterator<Item = &TopologyObject> + Clone + ExactSizeIterator + FusedIterator {
        self.singly_linked_children(self.memory_first_child, self.memory_arity())
    }

    /// Total memory (in bytes) in NUMA nodes below this object
    ///
    /// Requires [`DiscoverySupport::numa_memory()`].
    pub fn total_memory(&self) -> u64 {
        self.total_memory
    }

    /// Number of I/O children.
    pub fn io_arity(&self) -> u32 {
        self.io_arity
    }

    /// I/O children of this object
    ///
    /// Bridges, PCI and OS devices are listed here instead of in the
    /// [`TopologyObject::normal_children()`] list. See also
    /// [`ObjectType::is_io()`].
    pub fn io_children(
        &self,
    ) -> impl Iterator<Item = &TopologyObject> + Clone + ExactSizeIterator + FusedIterator {
        self.singly_linked_children(self.io_first_child, self.io_arity())
    }

    /// Truth that this is a bridge covering the specified PCI bus
    #[doc(alias = "hwloc_bridge_covers_pcibus")]
    pub fn is_bridge_covering_pci_bus(&self, domain: PCIDomain, bus_id: u8) -> bool {
        let Some(ObjectAttributes::Bridge(bridge)) = self.attributes() else { return false };
        let Some(DownstreamAttributes::PCI(pci)) = bridge.downstream_attributes() else { return false };
        pci.domain() == domain && pci.secondary_bus() <= bus_id && pci.subordinate_bus() >= bus_id
    }

    /// Number of Misc children.
    pub fn misc_arity(&self) -> u32 {
        self.misc_arity
    }

    /// Misc children of this object
    ///
    /// Misc objects are listed here instead of in the
    /// [`TopologyObject::normal_children()`] list.
    pub fn misc_children(
        &self,
    ) -> impl Iterator<Item = &TopologyObject> + Clone + ExactSizeIterator + FusedIterator {
        self.singly_linked_children(self.misc_first_child, self.misc_arity())
    }

    /// Full list of children (normal, then memory, then I/O, then Misc)
    pub fn all_children(&self) -> impl Iterator<Item = &TopologyObject> + Clone + FusedIterator {
        self.normal_children()
            .chain(self.memory_children())
            .chain(self.io_children())
            .chain(self.misc_children())
    }

    /// Iterator over singly linked lists of child TopologyObjects with arity
    fn singly_linked_children(
        &self,
        first: *mut TopologyObject,
        arity: u32,
    ) -> impl Iterator<Item = &TopologyObject> + Clone + ExactSizeIterator + FusedIterator {
        let mut current = first;
        (0..arity).map(move |_| {
            assert!(!current.is_null(), "Got null child before expected arity");
            let result = unsafe { &*current };
            current = result.next_sibling;
            result
        })
    }
}

/// # CPU set
impl TopologyObject {
    /// CPUs covered by this object.
    ///
    /// This is the set of CPUs for which there are PU objects in the
    /// topology under this object, i.e. which are known to be physically
    /// contained in this object and known how (the children path between this
    /// object and the PU objects).
    ///
    /// If the [`BuildFlags::INCLUDE_DISALLOWED`] topology building
    /// configuration flag is set, some of these CPUs may be online but not
    /// allowed for binding, see [`Topology::allowed_cpuset()`].
    ///
    /// All objects have CPU and node sets except Misc and I/O objects.
    pub fn cpuset(&self) -> Option<&CpuSet> {
        unsafe { CpuSet::borrow_from_raw_mut(&self.cpuset) }
    }

    /// Truth that this object is inside of the given cpuset `set`
    ///
    /// Objects are considered to be inside `set` if they have a non-empty
    /// cpuset which verifies `set.includes(object_cpuset)`
    pub fn is_inside_cpuset(&self, set: &CpuSet) -> bool {
        let Some(object_cpuset) = self.cpuset() else { return false };
        set.includes(object_cpuset) && !object_cpuset.is_empty()
    }

    /// Truth that this object covers the given cpuset `set`
    ///
    /// Objects are considered to cover `set` if it is non-empty and the object
    /// has a cpuset which verifies `object_cpuset.includes(set)
    pub fn covers_cpuset(&self, set: &CpuSet) -> bool {
        let Some(object_cpuset) = self.cpuset() else { return false };
        object_cpuset.includes(set) && !set.is_empty()
    }

    /// The complete CPU set of logical processors of this object.
    ///
    /// This includes not only the same as the cpuset field, but also the
    /// CPUs for which topology information is unknown or incomplete, some
    /// offline CPUs, and the CPUs that are ignored when the
    /// [`BuildFlags::INCLUDE_DISALLOWED`] topology building configuration flag
    /// is not set.
    ///
    /// Thus no corresponding PU object may be found in the topology, because
    /// the precise position is undefined. It is however known that it would be
    /// somewhere under this object.
    pub fn complete_cpuset(&self) -> Option<&CpuSet> {
        unsafe { CpuSet::borrow_from_raw_mut(&self.complete_cpuset) }
    }
}

/// # NUMA node set
impl TopologyObject {
    /// NUMA nodes covered by this object or containing this object.
    ///
    /// This is the set of NUMA nodes for which there are NODE objects in the
    /// topology under or above this object, i.e. which are known to be
    /// physically contained in this object or containing it and known how (the
    /// children path between this object and the NODE objects). In the end,
    /// these nodes are those that are close to the current object.
    ///
    #[cfg_attr(
        feature = "hwloc-2_3_0",
        doc = "With hwloc 2.3+, [`Topology::local_numa_nodes()`] may be used to list those NUMA nodes"
    )]
    #[cfg_attr(feature = "hwloc-2_3_0", doc = "more precisely.")]
    ///
    /// If the [`BuildFlags::INCLUDE_DISALLOWED`] topology building
    /// configuration flag is set, some of these nodes may not be allowed for
    /// allocation, see [`Topology::allowed_nodeset()`].
    ///
    /// If there are no NUMA nodes in the machine, all the memory is close to
    /// this object, so the nodeset is full.
    ///
    /// All objects have CPU and node sets except Misc and I/O objects.
    pub fn nodeset(&self) -> Option<&NodeSet> {
        unsafe { NodeSet::borrow_from_raw_mut(&self.nodeset) }
    }

    /// The complete NUMA node set of this object,.
    ///
    /// This includes not only the same as the nodeset field, but also the NUMA
    /// nodes for which topology information is unknown or incomplete, some
    /// offline nodes, and the nodes that are ignored when the
    /// [`BuildFlags::INCLUDE_DISALLOWED`] topology building configuration flag
    /// is not set.
    ///
    /// Thus no corresponding NUMANode object may be found in the topology,
    /// because the precise position is undefined. It is however known that it
    /// would be somewhere under this object.
    ///
    /// If there are no NUMA nodes in the machine, all the memory is close to
    /// this object, so complete_nodeset is full.
    pub fn complete_nodeset(&self) -> Option<&NodeSet> {
        unsafe { NodeSet::borrow_from_raw_mut(&self.complete_nodeset) }
    }
}

/// # Key-value information
impl TopologyObject {
    /// Complete list of (key, value) textual info pairs
    ///
    /// hwloc defines a number of standard object info attribute names with
    /// associated semantics, please check out
    /// <https://hwloc.readthedocs.io/en/v2.9/attributes.html#attributes_info>
    /// for more information.
    pub fn infos(&self) -> &[TextualInfo] {
        if self.children.is_null() {
            assert_eq!(
                self.infos_count, 0,
                "Got null infos pointer with nonzero info count"
            );
            return &[];
        }

        unsafe {
            std::slice::from_raw_parts(
                self.infos,
                // If this fails, it means infos_count does not fit in a
                // size_t, but by definition of size_t that cannot happen...
                usize::try_from(self.infos_count).expect("Should not happen"),
            )
        }
    }

    /// Search the given key name in object infos and return the corresponding value
    ///
    /// If multiple keys match the given name, only the first one is returned.
    ///
    /// Calling this operation multiple times will result in duplicate work. If
    /// you need to do this sort of search many times, you should collect
    /// `infos()` into a `HashMap` or `BTreeMap` for increased lookup efficiency.
    pub fn info(&self, key: &str) -> Option<&CStr> {
        self.infos().iter().find_map(|info| {
            let Ok(info_name) = info.name().to_str() else { return None };
            (info_name == key).then_some(info.value())
        })
    }

    /// Add the given info name and value pair to the given object
    ///
    /// The info is appended to the existing info array even if another key with
    /// the same name already exists.
    ///
    /// The input strings are copied before being added in the object infos.
    ///
    /// This function may be used to enforce object colors in the lstopo
    /// graphical output by using "lstopoStyle" as a name and "Background=#rrggbb"
    /// as a value. See `CUSTOM COLORS` in the `lstopo(1)` manpage for details.
    ///
    /// If value contains some non-printable characters, they will be dropped
    /// when exporting to XML.
    ///
    /// # Errors
    ///
    /// - [`NulError`] if `name` or `value` contains NUL chars.
    pub fn add_info(&mut self, name: &str, value: &str) -> Result<(), NulError> {
        let name = LibcString::new(name)?;
        let value = LibcString::new(value)?;
        let result = unsafe { ffi::hwloc_obj_add_info(self, name.borrow(), value.borrow()) };
        assert_ne!(result, -1, "Failed to add info to object");
        assert_eq!(result, 0, "Unexpected result from hwloc_obj_add_info");
        Ok(())
    }
}

// # Internal utilities
impl TopologyObject {
    /// Display the TopologyObject's type and attributes
    fn display(&self, f: &mut fmt::Formatter, verbose: bool) -> fmt::Result {
        let type_chars = ffi::call_snprintf(|buf, len| unsafe {
            ffi::hwloc_obj_type_snprintf(buf, len, self, verbose.into())
        });

        let separator = if f.alternate() {
            b"\n  \0".as_ptr()
        } else {
            b"  \0".as_ptr()
        }
        .cast::<c_char>();
        let attr_chars = ffi::call_snprintf(|buf, len| unsafe {
            ffi::hwloc_obj_attr_snprintf(buf, len, self, separator, verbose.into())
        });

        unsafe {
            let type_str = CStr::from_ptr(type_chars.as_ptr()).to_string_lossy();
            let attr_str = CStr::from_ptr(attr_chars.as_ptr()).to_string_lossy();
            if attr_str.is_empty() {
                write!(f, "{type_str}")
            } else if f.alternate() {
                write!(f, "{type_str} (\n  {attr_str}\n)")
            } else {
                write!(f, "{type_str} ({attr_str})")
            }
        }
    }
}

impl fmt::Display for TopologyObject {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.display(f, false)
    }
}

impl fmt::Debug for TopologyObject {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.display(f, true)
    }
}

unsafe impl Send for TopologyObject {}
unsafe impl Sync for TopologyObject {}
