//! Looking for objects using non-hierarchical criteria

mod io;

use super::{types::ObjectType, TopologyObject, TopologyObjectID};
#[cfg(feature = "hwloc-2_5_0")]
use crate::errors::NulError;
#[cfg(doc)]
use crate::topology::support::DiscoverySupport;
use crate::{
    bitmap::BitmapRef,
    cpu::cpuset::CpuSet,
    errors::{ForeignObjectError, ParameterError},
    memory::nodeset::NodeSet,
    topology::Topology,
};
#[allow(unused)]
#[cfg(test)]
use similar_asserts::assert_eq;
use std::{fmt::Debug, iter::FusedIterator, ops::Deref, ptr};
use thiserror::Error;

/// # Finding other objects
//
// --- Implementation details ---
//
// This is inspired by the upstream functionality described at
// https://hwloc.readthedocs.io/en/v2.9/group__hwlocality__helper__find__misc.html
// but the code had to be ported to Rust because it's inline
impl Topology {
    /// Get the object of type [`ObjectType::PU`] with the specified OS index
    ///
    /// If you want to convert an entire CPU set into the PU objects it
    /// contains, using [`pus_from_cpuset()`] will be more efficient than
    /// repeatedly calling this function with every OS index from the [`CpuSet`].
    ///
    /// Requires [`DiscoverySupport::pu_count()`].
    ///
    /// [`pus_from_cpuset()`]: Self::pus_from_cpuset()
    #[doc(alias = "hwloc_get_pu_obj_by_os_index")]
    pub fn pu_with_os_index(&self, os_index: usize) -> Option<&TopologyObject> {
        self.objs_and_os_indices(ObjectType::PU)
            .find_map(|(pu, pu_os_index)| (pu_os_index == os_index).then_some(pu))
    }

    /// Get the objects of type [`ObjectType::PU`] covered by the specified cpuset
    ///
    /// `cpuset` can be a `&'_ CpuSet` or a `BitmapRef<'_, CpuSet>`.
    ///
    /// Requires [`DiscoverySupport::pu_count()`].
    ///
    /// This functionality is specific to the Rust bindings.
    pub fn pus_from_cpuset<'result>(
        &'result self,
        cpuset: impl Deref<Target = CpuSet> + Clone + 'result,
    ) -> impl DoubleEndedIterator<Item = &TopologyObject> + Clone + FusedIterator + 'result {
        self.objs_and_os_indices(ObjectType::PU)
            .filter_map(move |(pu, os_index)| cpuset.is_set(os_index).then_some(pu))
    }

    /// Get the object of type [`NUMANode`] with the specified OS index
    ///
    /// If you want to convert an entire [`NodeSet` into the [`NUMANode`]
    /// objects it contains, using [`nodes_from_nodeset()`] will be more
    /// efficient than repeatedly calling this function with every OS index from
    /// the [`NodeSet`].
    ///
    /// Requires [`DiscoverySupport::numa_count()`].
    ///
    /// [`nodes_from_nodeset()`]: Self::nodes_from_nodeset()
    /// [`NUMANode`]: ObjectType::NUMANode
    #[doc(alias = "hwloc_get_numanode_obj_by_os_index")]
    pub fn node_with_os_index(&self, os_index: usize) -> Option<&TopologyObject> {
        self.objs_and_os_indices(ObjectType::NUMANode)
            .find_map(|(node, node_os_index)| (node_os_index == os_index).then_some(node))
    }

    /// Get the objects of type [`ObjectType::NUMANode`] covered by the
    /// specified nodeset
    ///
    /// `nodeset` can be a `&'_ NodeSet` or a `BitmapRef<'_, NodeSet>`.
    ///
    /// Requires [`DiscoverySupport::numa_count()`].
    ///
    /// This functionality is specific to the Rust bindings.
    pub fn nodes_from_nodeset<'result>(
        &'result self,
        nodeset: impl Deref<Target = NodeSet> + Clone + 'result,
    ) -> impl DoubleEndedIterator<Item = &TopologyObject> + Clone + FusedIterator + 'result {
        self.objs_and_os_indices(ObjectType::NUMANode)
            .filter_map(move |(node, os_index)| nodeset.is_set(os_index).then_some(node))
    }

    /// Get a list of `(&TopologyObject, OS index)` tuples for an `ObjectType`
    /// that is guaranteed to appear only at one depth of the topology and to
    /// have an OS index.
    ///
    /// # Panics
    ///
    /// Will panic if the object type appears at more than one depth or do not
    /// have an OS index. As this method is an implementation detail of other
    /// methods above, the caller should be able to ensure it never happens.
    fn objs_and_os_indices(
        &self,
        ty: ObjectType,
    ) -> impl DoubleEndedIterator<Item = (&TopologyObject, usize)>
           + Clone
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
    /// topology tree).
    ///
    /// This search may only be applied to objects that have a cpuset (normal
    /// and memory objects) and belong to this topology.
    ///
    /// # Errors
    ///
    /// - [`ForeignObject`] if `obj` does not belong to this topology.
    /// - [`MissingCpuSet`] if `obj` does not have a cpuset.
    ///
    /// [`ForeignObject`]: ClosestObjectsError::ForeignObject
    /// [`MissingCpuSet`]: ClosestObjectsError::MissingCpuSet
    #[doc(alias = "hwloc_get_closest_objs")]
    pub fn objects_closest_to<'result>(
        &'result self,
        obj: &'result TopologyObject,
    ) -> Result<impl Iterator<Item = &TopologyObject> + Clone + 'result, ClosestObjectsError> {
        // Validate input object
        if !self.contains(obj) {
            return Err(ClosestObjectsError::ForeignObject(obj.into()));
        }
        let obj_cpuset = obj
            .cpuset()
            .ok_or_else(|| ClosestObjectsError::MissingCpuSet(obj.into()))?;

        /// Assert that an object has a cpuset, return both
        fn obj_and_cpuset<'obj>(
            obj: &'obj TopologyObject,
            error: &str,
        ) -> (&'obj TopologyObject, BitmapRef<'obj, CpuSet>) {
            (obj, obj.cpuset().expect(error))
        }

        /// Find the first ancestor of an object that knows about more objects
        /// than that object (if any), and return it along with its cpuset
        fn find_larger_parent<'obj>(
            known_obj: &'obj TopologyObject,
            known_cpuset: &CpuSet,
        ) -> Option<(&'obj TopologyObject, BitmapRef<'obj, CpuSet>)> {
            known_obj
                .ancestors()
                .map(|ancestor| {
                    obj_and_cpuset(
                        ancestor,
                        "Ancestors of an object with a cpuset should have a cpuset",
                    )
                })
                .find(|(_ancestor, ancestor_cpuset)| ancestor_cpuset != known_cpuset)
        }
        let mut ancestor_and_cpuset = find_larger_parent(obj, &obj_cpuset);

        // Prepare to jointly iterate over cousins and their cpusets
        // On each pass, we're going to find which cousins are covered by the
        // current ancestor, keeping the other cousins around to iterate over
        // them again during the next pass with a higher-level ancestor.
        let mut cousins_and_cpusets = self
            .objects_at_depth(obj.depth())
            .filter(|cousin_or_obj| !ptr::eq(*cousin_or_obj, obj))
            .map(|cousin| {
                obj_and_cpuset(
                    cousin,
                    "Cousins of an object with a cpuset should have a cpuset",
                )
            })
            .collect::<Vec<_>>();
        let mut next_cousins_and_cpusets = Vec::new();

        // Emit the final iterator
        Ok(std::iter::from_fn(move || {
            loop {
                // Look for a cousin that is covered by the current ancestor
                let (ancestor, ancestor_cpuset) = ancestor_and_cpuset.take()?;
                while let Some((cousin, cousin_cpuset)) = cousins_and_cpusets.pop() {
                    if ancestor_cpuset.includes(cousin_cpuset) {
                        ancestor_and_cpuset = Some((ancestor, ancestor_cpuset));
                        return Some(cousin);
                    } else {
                        next_cousins_and_cpusets.push((cousin, cousin_cpuset));
                    }
                }

                // We ran out of cousins, go to a higher-level ancestor or end
                // iteration if we reached the top of the tree.
                let (ancestor, ancestor_cpuset) = find_larger_parent(ancestor, &ancestor_cpuset)?;
                ancestor_and_cpuset = Some((ancestor, ancestor_cpuset));
                std::mem::swap(&mut cousins_and_cpusets, &mut next_cousins_and_cpusets);
            }
        }))
    }

    /// Find an object via a parent->child chain specified by types and indices
    ///
    /// For example, if called with `&[(NUMANode, 0), (Package, 1), (Core, 2)]`,
    /// this will return the third core object below the second package below
    /// the first NUMA node.
    ///
    /// The first object is indexed relative to the topology's root Machine
    /// object and searched amongst its children. As a consequence, the root
    /// Machine object cannot be found using this method.
    ///
    /// This search may only be applied to object types that have a cpuset
    /// (normal and memory objects).
    ///
    /// # Errors
    ///
    /// - [`ParameterError`] if one of the specified object types does not have
    ///   a cpuset.
    #[doc(alias = "hwloc_get_obj_below_array_by_type")]
    #[doc(alias = "hwloc_get_obj_below_by_type")]
    pub fn object_by_type_index_path(
        &self,
        path: &[(ObjectType, usize)],
    ) -> Result<Option<&TopologyObject>, ParameterError<ObjectType>> {
        // Make sure the path only includes object types with cpusets
        if let Some(&(bad_ty, _idx)) = path.iter().find(|(ty, _idx)| !ty.has_sets()) {
            return Err(ParameterError::from(bad_ty));
        }

        // Then perform the actual search
        let mut subroot = self.root_object();
        for &(ty, idx) in path {
            // We checked the presence of a subroot cpuset in the beginning
            let cpuset = subroot
                .cpuset()
                .expect("subroot should have a cpuset per above check");

            // Define what it means to be a child
            let is_child =
                |obj: &&TopologyObject| obj.ancestors().any(|ancestor| ptr::eq(ancestor, subroot));

            // Look up child efficienty using cpuset
            if let Some(next_obj) = self
                .objects_inside_cpuset_with_type(cpuset, ty)
                .nth(idx)
                .filter(is_child)
            {
                subroot = next_obj;
            } else {
                return Ok(None);
            }
        }
        Ok(Some(subroot))
    }

    /// Find an object of a different type with the same locality
    ///
    /// The source object `src` must belong to this topology, otherwise a
    /// [`ForeignSource`] error will be returned.
    ///
    /// If the source object is a normal or memory type, this function returns
    /// an object of type `ty` with the same CPU and node sets, either below or
    /// above in the hierarchy.
    ///
    /// If the source object is a PCI or an OS device within a PCI device, the
    /// function may either return that PCI device, or another OS device in the
    /// same PCI parent. This may for instance be useful for converting between
    /// OS devices such as "nvml0" or "rsmi1" used in distance structures into
    /// the the PCI device, or the CUDA or OpenCL OS device that correspond to
    /// the same physical card.
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
    /// - [`ForeignSource`] if `src` does not belong to this topology.
    /// - [`IncompatibleTypes`] if `src` is a normal/memory object and `ty` is
    ///   an I/O or Misc object type, or vice versa.
    /// - [`StringContainsNul`] if `subtype` or `name_prefix` contains NUL chars.
    ///
    /// [`ForeignSource`]: LocalObjectError::ForeignSource
    /// [`IncompatibleTypes`]: LocalObjectError::IncompatibleTypes
    /// [`StringContainsNul`]: LocalObjectError::StringContainsNul
    #[cfg(feature = "hwloc-2_5_0")]
    #[doc(alias = "hwloc_get_obj_with_same_locality")]
    pub fn object_with_same_locality(
        &self,
        src: &TopologyObject,
        ty: ObjectType,
        subtype: Option<&str>,
        name_prefix: Option<&str>,
    ) -> Result<Option<&TopologyObject>, LocalObjectError> {
        use crate::ffi::{string::LibcString, transparent::AsNewtype};
        use std::ffi::c_char;
        if !self.contains(src) {
            return Err(src.into());
        }
        let src_ty = src.object_type();
        if src_ty.has_sets() ^ ty.has_sets() {
            return Err(LocalObjectError::IncompatibleTypes(src_ty, ty));
        }
        let subtype = subtype.map(LibcString::new).transpose()?;
        let name_prefix = name_prefix.map(LibcString::new).transpose()?;
        let borrow_pchar = |opt: &Option<LibcString>| -> *const c_char {
            opt.as_ref().map_or(ptr::null(), LibcString::borrow)
        };
        // SAFETY: - Topology is trusted to contain a valid ptr (type invariant)
        //         - src was checked to belong to the active topology
        //         - LibcStrings are trusted to be valid C strings and not used
        //           after the end of their lifetime
        //         - hwloc ops are trusted not to modify *const parameters
        //         - By construction, ObjectType only exposes values that map into
        //           hwloc_obj_type_t values understood by the configured version
        //           of hwloc, and build.rs checks that the active version of
        //           hwloc is not older than that, so into() may only generate
        //           valid hwloc_obj_type_t values for current hwloc
        //         - Per documentation, flags must be zero
        let ptr = unsafe {
            hwlocality_sys::hwloc_get_obj_with_same_locality(
                self.as_ptr(),
                &src.0,
                ty.into(),
                borrow_pchar(&subtype),
                borrow_pchar(&name_prefix),
                0,
            )
        };
        // SAFETY: - If hwloc succeeds, the output pointer and its target are
        //           both assumed to be valid
        //         - Output is bound to the lifetime of the topology it comes from
        Ok((!ptr.is_null()).then(|| unsafe { (&*ptr).as_newtype() }))
    }
}

/// Error returned by [`Topology::objects_closest_to()`]
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum ClosestObjectsError {
    /// Target object does not belong to this topology
    #[error(transparent)]
    ForeignObject(#[from] ForeignObjectError),

    /// Target object does not have a cpuset and this search requires one
    #[error(transparent)]
    MissingCpuSet(#[from] MissingObjCpuSetError),
}

/// Error returned when a search algorithm that requires a cpuset is applied to
/// an object that doesn't have one.
///
/// The presence of a cpuset greatly simplifies some search algorithms as it
/// allows asserting that an object is a child of another with simple bitmap
/// operations, rather than requiring topology tree traversal. Therefore,
/// relatively complex search operations may only be applied to objects with a
/// cpuset (i.e. normal and memory objects) and will fail with this error if
/// applied to other object types.
//
// --- Implementation notes ---
//
// Not implementing Copy at this point because I want to leave options open for
// switching to another way to describe objects (Debug string, etc).
#[allow(missing_copy_implementations)]
#[derive(Clone, Debug, Default, Eq, Error, PartialEq)]
#[error("object #{0} doesn't have a cpuset but we need one for this search")]
pub struct MissingObjCpuSetError(TopologyObjectID);
//
impl<'topology> From<&'topology TopologyObject> for MissingObjCpuSetError {
    fn from(object: &'topology TopologyObject) -> Self {
        Self(object.global_persistent_index())
    }
}

/// Error returned by [`Topology::object_with_same_locality()`]
#[cfg(feature = "hwloc-2_5_0")]
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum LocalObjectError {
    /// Target object does not belong to this topology
    #[error(transparent)]
    ForeignSource(#[from] ForeignObjectError),

    /// Source object is a normal/memory object and target type is an I/O or
    /// Misc type, or vice-versa
    #[error("source object type {0} and destination object type {1} are incompatible")]
    IncompatibleTypes(ObjectType, ObjectType),

    /// Subtype or name prefix string contains a NUL char
    #[error("local object query string can't contain NUL chars")]
    StringContainsNul,
}
//
#[cfg(feature = "hwloc-2_5_0")]
impl From<NulError> for LocalObjectError {
    fn from(_: NulError) -> Self {
        Self::StringContainsNul
    }
}
//
#[cfg(feature = "hwloc-2_5_0")]
impl<'topology> From<&'topology TopologyObject> for LocalObjectError {
    fn from(object: &'topology TopologyObject) -> Self {
        Self::ForeignSource(object.into())
    }
}

#[allow(clippy::cognitive_complexity)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategies::{any_object, topology_related_set};
    use proptest::prelude::*;
    use std::{
        collections::{BTreeMap, HashMap},
        sync::OnceLock,
    };

    // Tests that should only be run on a subset of objects because the property
    // of interest can only be evaluated by traversing the whole topology or a
    // major subset thereof.
    proptest! {
        /// Test that [`Topology::objects_closest_to()`] works as expected
        #[test]
        fn objects_closest_to(obj in any_object()) {
            // Invoke the query
            let topology = Topology::test_instance();
            let result = topology.objects_closest_to(obj);

            // Calling it on a foreign object is an error
            if !topology.contains(obj) {
                return Ok(prop_assert!(matches!(
                    result,
                    Err(ClosestObjectsError::ForeignObject(e))
                        if e == ForeignObjectError::from(obj)
                )));
            }

            // Calling it on an object without a cpuset is also an error
            if obj.cpuset().is_none() {
                return Ok(prop_assert!(matches!(
                    result,
                    Err(ClosestObjectsError::MissingCpuSet(e))
                        if e == MissingObjCpuSetError::from(obj)
                )));
            }

            // Build a full list of this object's cousins, sorted by common
            // ancestor distance and keyed by global persistent index
            let cousin_iter = |next: fn(&TopologyObject) -> Option<&TopologyObject>| {
                std::iter::successors(next(obj), move |cousin| next(cousin))
            };
            let all_cousins =
                cousin_iter(TopologyObject::prev_cousin)
                    .chain(cousin_iter(TopologyObject::next_cousin));
            let mut cousins_by_distance = BTreeMap::<usize, HashMap<u64, &TopologyObject>>::new();
            for cousin in all_cousins {
                let common_ancestor = obj.first_common_ancestor(cousin).unwrap();
                let ancestor_distance = obj
                    .ancestors()
                    .position(|ancestor| ptr::eq(ancestor, common_ancestor))
                    .unwrap();
                let already_seen = cousins_by_distance.entry(ancestor_distance).or_default().insert(cousin.global_persistent_index(), cousin);
                prop_assert!(already_seen.is_none());
            }
            let max_distance = cousins_by_distance.last_entry().map_or(0, |entry| *entry.key());

            // Check if iterator matches this expectation
            let mut iterator = result.unwrap();
            'distance: for (distance, mut cousin_set) in cousins_by_distance {
                // Continue iteration over neighbors, continue outer loop when
                // all cousins at this distance have been exhausted
                for neighbor in iterator.by_ref() {
                    // Check that the proposed neighbor is a cousin at the
                    // expected distance: closest cousins should come first
                    prop_assert!(
                        cousin_set.remove(&neighbor.global_persistent_index()).is_some()
                    );

                    // We reached the last cousin at this distance, move to the
                    // next cousin distance.
                    if cousin_set.is_empty() {
                        continue 'distance;
                    }
                }

                // Iteration can only end when all cousins have been seen
                prop_assert_eq!(distance, max_distance);
                prop_assert!(cousin_set.is_empty());
            }
            // Iteration should end once all cousins have been seen
            prop_assert!(iterator.next().is_none());
        }
    }

    // --- Querying stuff by OS index ---

    /// Build an OS index -> [`TopologyObject`] mapping for some [`ObjectType`]
    ///
    /// Only call this for object types which are guaranteed to have a unique OS
    /// index, namely PUs and NUMA nodes.
    fn os_index_to_object(ty: ObjectType) -> HashMap<usize, &'static TopologyObject> {
        let topology = Topology::test_instance();
        let mut map = HashMap::new();
        for pu in topology.objects_with_type(ty) {
            assert!(map.insert(pu.os_index().unwrap(), pu).is_none());
        }
        map
    }

    /// Check one of the `Topology::xyz_with_os_index` functions
    fn check_object_with_os_index(
        method: impl FnOnce(&Topology, usize) -> Option<&TopologyObject>,
        os_index: usize,
        expected: &HashMap<usize, &TopologyObject>,
    ) {
        let topology = Topology::test_instance();
        match (method(topology, os_index), expected.get(&os_index)) {
            (Some(obj1), Some(obj2)) if ptr::eq(obj1, *obj2) => {}
            (None, None) => {}
            other => panic!("unequected pu_with_os_index result vs expectation: {other:?}"),
        }
    }

    /// OS index -> PU mapping
    fn os_index_to_pu() -> &'static HashMap<usize, &'static TopologyObject> {
        static MAP: OnceLock<HashMap<usize, &'static TopologyObject>> = OnceLock::new();
        MAP.get_or_init(|| os_index_to_object(ObjectType::PU))
    }

    /// Test [`Topology::pu_with_os_index()`]
    fn check_pu_with_os_index(os_index: usize) {
        check_object_with_os_index(Topology::pu_with_os_index, os_index, os_index_to_pu());
    }

    /// Exhaustive check for all valid PU OS indices
    #[test]
    fn valid_pu_with_os_index() {
        for os_index in os_index_to_pu().keys() {
            check_pu_with_os_index(*os_index);
        }
    }

    proptest! {
        /// Stochastic test for possibly-nonexistent PU OS indices
        fn any_pu_with_os_index(os_index: usize) {
            check_pu_with_os_index(os_index)
        }
    }

    /// OS index -> NUMA node mapping
    fn os_index_to_node() -> &'static HashMap<usize, &'static TopologyObject> {
        static MAP: OnceLock<HashMap<usize, &'static TopologyObject>> = OnceLock::new();
        MAP.get_or_init(|| os_index_to_object(ObjectType::NUMANode))
    }

    /// Test [`Topology::pu_with_os_index()`]
    fn check_node_with_os_index(os_index: usize) {
        check_object_with_os_index(Topology::node_with_os_index, os_index, os_index_to_node());
    }

    /// Exhaustive check for all valid NUMA node OS indices
    #[test]
    fn valid_node_with_os_index() {
        for os_index in os_index_to_node().keys() {
            check_node_with_os_index(*os_index);
        }
    }

    proptest! {
        /// Stochastic test for possibly-nonexistent NUMA node OS indices
        fn any_node_with_os_index(os_index: usize) {
            check_node_with_os_index(os_index)
        }
    }

    // --- Querying stuff by cpuset/nodeset ---

    proptest! {
        /// Test [`Topology::pus_from_cpuset()`]
        #[test]
        fn pus_from_cpuset(cpuset in topology_related_set(Topology::cpuset)) {
            let mut expected = os_index_to_pu().clone();
            expected.retain(|_idx, pu| {
                cpuset.includes(pu.cpuset().unwrap())
            });

            let topology = Topology::test_instance();
            let actual = topology.pus_from_cpuset(&cpuset);

            prop_assert_eq!(actual.clone().count(), expected.len());
            for pu in actual {
                prop_assert!(expected.contains_key(&pu.os_index().unwrap()));
            }
        }

        /// Test [`Topology::nodes_from_nodeset()`]
        #[test]
        fn nodes_from_nodeset(nodeset in topology_related_set(Topology::nodeset)) {
            let mut expected = os_index_to_node().clone();
            expected.retain(|_idx, node| {
                nodeset.includes(node.nodeset().unwrap())
            });

            let topology = Topology::test_instance();
            let actual = topology.nodes_from_nodeset(&nodeset);

            prop_assert_eq!(actual.clone().count(), expected.len());
            for node in actual {
                prop_assert!(expected.contains_key(&node.os_index().unwrap()));
            }
        }
    }

    // --- Querying objects by type-index path ---

    /// Generate an object that has a cpuset
    fn object_with_cpuset() -> impl Strategy<Value = &'static TopologyObject> + Clone {
        let objects_with_cpusets = Topology::test_objects()
            .iter()
            .copied()
            .filter(|obj| obj.cpuset().is_some())
            .collect::<Vec<_>>();
        prop::sample::select(objects_with_cpusets)
    }

    /// Generate type-index paths that are mostly valid, but will occasionally
    /// be disordered or invalid, tell what the expected result is if the path
    /// is valid and None otherwise
    fn type_index_path(
    ) -> impl Strategy<Value = (Vec<(ObjectType, usize)>, Option<&'static TopologyObject>)> {
        // First, have a strategy for generating correct paths
        let topology = Topology::test_instance();
        let valid_path = object_with_cpuset()
            .prop_filter("Machine can't be found using by type_index_path", |obj| {
                obj.object_type() != ObjectType::Machine
            })
            .prop_flat_map(move |obj| {
                // Start by generating a valid object ancestor path, excluding the
                // topology's root object
                let mut full_ancestor_path = obj.ancestors().collect::<Vec<_>>();
                full_ancestor_path.pop();
                full_ancestor_path.reverse();
                let num_ancestors = full_ancestor_path.len();

                // Extract a subsequence of it
                prop::sample::subsequence(full_ancestor_path, 0..=num_ancestors).prop_map(
                    move |mut obj_path| {
                        // Append the object at the end to make it a full path
                        obj_path.push(obj);

                        // Convert the path to type -> index form
                        let mut last_parent = topology.root_object();
                        let type_index_path = obj_path
                            .into_iter()
                            .map(|obj| {
                                let ty = obj.object_type();
                                let index = topology
                                    .objects_inside_cpuset_with_type(
                                        last_parent.cpuset().unwrap(),
                                        ty,
                                    )
                                    .position(|candidate| ptr::eq(candidate, obj))
                                    .unwrap();
                                last_parent = obj;
                                (ty, index)
                            })
                            .collect::<Vec<_>>();
                        (type_index_path, Some(obj))
                    },
                )
            });

        // Order matters, so a path in the wrong order is not a valid path
        let disordered_path = valid_path.clone().prop_flat_map(|(path, obj)| {
            let ordered_path = path.clone();
            Just(path).prop_shuffle().prop_map(move |shuffled_path| {
                let disordered = shuffled_path != ordered_path;
                (shuffled_path, (!disordered).then(|| obj.unwrap()))
            })
        });

        // Random paths are most likely wrong, but could be right sometimes
        let random_path = any::<Vec<(ObjectType, usize)>>().prop_map(move |path| {
            // The root must not appear in the path
            if path.first().map(|(ty, _)| ty) == Some(&ObjectType::Machine) {
                return (path, None);
            }

            // Otherwise, we can search
            let mut last_parent = topology.root_object();
            for &(ty, idx) in &path {
                let is_child = |obj: &&TopologyObject| {
                    obj.ancestors()
                        .any(|ancestor| ptr::eq(ancestor, last_parent))
                };
                if let Some(obj) = topology
                    .objects_inside_cpuset_with_type(last_parent.cpuset().unwrap(), ty)
                    .nth(idx)
                    .filter(is_child)
                {
                    last_parent = obj;
                } else {
                    return (path, None);
                }
            }
            (path, Some(last_parent))
        });

        // Put it all together, biased towards the valid case
        prop_oneof![
            3 => valid_path,
            1 => disordered_path,
            1 => random_path,
        ]
    }

    proptest! {
        /// Test for [`Topology::object_by_type_index_path()`]
        #[test]
        fn object_by_type_index_path((path, obj) in type_index_path()) {
            // Perform the query
            let topology = Topology::test_instance();
            let result = topology.object_by_type_index_path(&path);

            // Check error handling for lack of cpuset
            for (ty, _) in path {
                if !ty.has_sets() {
                    prop_assert!(obj.is_none());
                    prop_assert!(matches!(
                        &result,
                        Err(e) if *e == ParameterError::from(ty)
                    ));
                    return Ok(());
                }
            }

            // Check normal search path
            match (result.unwrap(), obj) {
                (Some(actual), Some(expected)) => prop_assert!(ptr::eq(actual, expected)),
                (None, None) => {}
                other => prop_assert!(false, "result/expectation mismatch: {other:?}"),
            }
        }
    }

    // --- Finding more objects with the same locality ---

    #[cfg(feature = "hwloc-2_5_0")]
    mod object_with_same_locality {
        use super::*;
        use crate::strategies::any_string;
        use std::{collections::HashSet, ffi::CStr};

        /// Lists of subtypes and names in the test topology
        fn subtypes_and_name_prefixes() -> (HashSet<&'static str>, HashSet<String>) {
            fn object_strings(
                mut get_string: impl FnMut(&TopologyObject) -> Option<&CStr>,
            ) -> HashSet<&'static str> {
                Topology::test_objects()
                    .iter()
                    .filter_map(|obj| get_string(obj))
                    .flat_map(CStr::to_str)
                    .collect()
            }
            let subtypes = object_strings(TopologyObject::subtype);
            let names = object_strings(TopologyObject::name);
            let name_prefixes = names
                .into_iter()
                .flat_map(|name| {
                    (1..=name.chars().count())
                        .map(|num_chars| name.chars().take(num_chars).collect::<String>())
                })
                .chain(std::iter::once(String::new()))
                .collect();
            (subtypes, name_prefixes)
        }

        /// Generate a subtype and name prefix input for
        /// [`Topology::object_with_same_locality()`]
        fn subtype_and_name_prefix() -> impl Strategy<Value = (Option<String>, Option<String>)> {
            let (subtypes, name_prefixes) = subtypes_and_name_prefixes();
            let subtypes = subtypes.into_iter().collect::<Vec<&'static str>>();
            let name_prefixes = name_prefixes.into_iter().collect::<Vec<String>>();
            let random_string = prop_oneof![any_string().prop_map(Some), Just(None),];
            let subtype = if subtypes.is_empty() {
                random_string.clone().boxed()
            } else {
                prop_oneof![
                    3 => prop::sample::select(subtypes).prop_map(|s| Some(s.to_owned())),
                    2 => random_string.clone(),
                ]
                .boxed()
            };
            let name_prefix = if name_prefixes.is_empty() {
                random_string.boxed()
            } else {
                prop_oneof![
                    3 => prop::sample::select(name_prefixes).prop_map(Some),
                    2 => random_string,
                ]
                .boxed()
            };
            (subtype, name_prefix)
        }

        proptest! {
            /// Test for [`Topology::object_with_same_locality()`]
            #[test]
            fn object_with_same_locality(
                src in any_object(),
                ty: ObjectType,
                (subtype, name_prefix) in subtype_and_name_prefix(),
            ) {
                // Start by running the method
                let topology = Topology::test_instance();
                let subtype = subtype.as_deref();
                let name_prefix = name_prefix.as_deref();
                let result = topology.object_with_same_locality(
                    src,
                    ty,
                    subtype,
                    name_prefix,
                );

                // Handle error cases
                if handle_error_cases(src, ty, subtype, name_prefix, &result)? {
                    return Ok(());
                }

                // These are the only two error conditions, otherwise the search
                // can fail but will always return an Option.
                match result.unwrap() {
                    Some(dst) => {
                        // Successful search should match all criteria
                        prop_assert!(topology.contains(dst));
                        prop_assert_eq!(dst.object_type(), ty);
                        if let Some(expected_subtype) = subtype {
                            prop_assert_eq!(dst.subtype().unwrap().to_str().unwrap(), expected_subtype);
                        }
                        if let Some(expected_prefix) = name_prefix {
                            prop_assert!(dst.name().unwrap().to_str().unwrap().starts_with(expected_prefix));
                        }
                        if ty.has_sets() {
                            prop_assert_eq!(dst.cpuset(), src.cpuset());
                            prop_assert_eq!(dst.nodeset(), src.nodeset());
                        } else if is_supported_io_type(ty) {
                            prop_assert!(is_io_local(src, dst)?);
                        }
                    }
                    None => {
                        // Search can fail only if no object in the topology
                        // would match all search criteria
                        'search: for obj in Topology::test_objects() {
                            if obj.object_type() != ty {
                                continue 'search;
                            }
                            if let Some(req_subtype) = subtype {
                                if let Some(obj_subtype) = obj.subtype().and_then(|cs| cs.to_str().ok()) {
                                    if req_subtype != obj_subtype {
                                        continue 'search;
                                    }
                                } else {
                                    continue 'search;
                                }
                            }
                            if let Some(name_prefix) = name_prefix {
                                if let Some(name) = obj.name().and_then(|cs| cs.to_str().ok()) {
                                    if !name.starts_with(name_prefix) {
                                        continue 'search;
                                    }
                                } else {
                                    continue 'search;
                                }
                            }
                            if ty.has_sets() {
                                prop_assert!((obj.cpuset() != src.cpuset()) || (obj.nodeset() != src.nodeset()));
                            } else if is_supported_io_type(ty) {
                                prop_assert!(!is_io_local(src, obj)?);
                            }
                        }
                    }
                }
            }
        }

        /// Handle error cases of [`Topology::object_with_same_locality()`],
        /// return truth that an error case was handled
        fn handle_error_cases(
            src: &TopologyObject,
            ty: ObjectType,
            subtype: Option<&str>,
            name_prefix: Option<&str>,
            result: &Result<Option<&TopologyObject>, LocalObjectError>,
        ) -> Result<bool, TestCaseError> {
            // Foreign objects should be reported as an error
            let topology = Topology::test_instance();
            if !topology.contains(src) {
                prop_assert!(matches!(result, Err(e) if e == &LocalObjectError::from(src)));
                return Ok(true);
            }

            // Converting between normal/memory object types and I/O types
            // or Misc is not allowed
            let src_ty = src.object_type();
            if src_ty.has_sets() ^ ty.has_sets() {
                prop_assert!(matches!(
                    result,
                    Err(LocalObjectError::IncompatibleTypes(src, dst)) if *src == src_ty && *dst == ty
                ));
                return Ok(true);
            }

            // NUL in input strings should be reported as an error
            for s in subtype.as_ref().into_iter().chain(name_prefix.as_ref()) {
                if s.contains('\0') {
                    prop_assert!(
                        matches!(result, Err(e) if e == &LocalObjectError::from(NulError))
                    );
                    return Ok(true);
                }
            }
            Ok(false)
        }

        /// Supported I/O object types for
        /// [`Topology::object_with_same_locality()`]
        fn is_supported_io_type(ty: ObjectType) -> bool {
            ty == ObjectType::PCIDevice || ty == ObjectType::OSDevice
        }

        /// Truth that `dst` is another I/O object below the same PCI bridge as
        /// `src`
        fn is_io_local(src: &TopologyObject, dst: &TopologyObject) -> Result<bool, TestCaseError> {
            // First, both src and dst must be OS or PCI devices
            if !(is_supported_io_type(src.object_type()) && is_supported_io_type(dst.object_type()))
            {
                return Ok(false);
            }

            // Find the first non-OS-device ancestor (normally a PCI device, but
            // may be something else if PCI device detection is disabled)
            let expected_parent = std::iter::once(src)
                .chain(src.ancestors())
                .find(|obj| obj.object_type() != ObjectType::OSDevice)
                .expect("OS devices should have a parent (at least Machine)");

            // dst may be either that PCI parent...
            if ptr::eq(dst, expected_parent) {
                return Ok(true);
            }

            // ...or an OS device below the same parent
            if ptr::eq(
                dst.parent().expect("OS devices should have a parent"),
                expected_parent,
            ) {
                prop_assert_eq!(dst.object_type(), ObjectType::OSDevice);
                return Ok(true);
            }
            Ok(false)
        }
    }
}
