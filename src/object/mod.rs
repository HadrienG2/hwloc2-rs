//! Objects within a hardware topology
//!
//! A [`Topology`] is first and foremost a tree of [`TopologyObject`] which
//! represents resource sharing relationships in hardware: an object is
//! considered the parent of all other objects that share the
//! most direct/fastest/lowest-latency route to it. For example, on x86, an L3
//! cache is the parent of a number of L2 caches, each the parent of one L1
//! cache, which is in turn the parent of a CPU core that may or may not be
//! shared by multiple hyperthreads (PUs in hwloc's vocabulary).
//!
//! This module defines the (very extensive) API through which one can query
//! various properties of topology objects and jump from them to other elements
//! of the surrounding topology.

pub mod attributes;
pub mod depth;
pub mod distance;
pub(crate) mod hierarchy;
pub(crate) mod lists;
pub mod search;
pub mod types;

use self::{
    attributes::{DownstreamAttributes, ObjectAttributes, PCIDomain},
    depth::{Depth, NormalDepth},
    types::ObjectType,
};
#[cfg(doc)]
use crate::topology::{builder::BuildFlags, support::DiscoverySupport, Topology};
use crate::{
    bitmap::BitmapRef,
    cpu::cpuset::CpuSet,
    ffi::{
        self, int,
        transparent::{AsNewtype, TransparentNewtype},
    },
    info::TextualInfo,
    memory::nodeset::NodeSet,
};
#[cfg(feature = "hwloc-2_3_0")]
use crate::{
    errors::{self, HybridError, NulError},
    ffi::string::LibcString,
};
use hwlocality_sys::{hwloc_obj, HWLOC_UNKNOWN_INDEX};
#[allow(unused)]
#[cfg(test)]
use similar_asserts::assert_eq;
use std::{
    ffi::{c_char, CStr},
    fmt::{self, Debug, Display},
    iter::FusedIterator,
    ops::Deref,
    ptr,
};

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
///
/// You cannot create an owned object of this type, it belongs to the topology.
//
// --- Implementation details ---
//
// Upstream docs:
// - https://hwloc.readthedocs.io/en/v2.9/structhwloc__obj.html
// - https://hwloc.readthedocs.io/en/v2.9/attributes.html
//
// See the matching accessor methods and hwloc documentation for more details on
// field semantics, the struct member documentation will only be focused on
// allowed interactions from methods.
//
// # Safety
//
// As a type invariant, all inner pointers are assumed to be safe to dereference
// and devoid of mutable aliases if the TopologyObject is reachable at all.
//
// This is enforced through the following precautions:
//
// - No API exposes an owned TopologyObjects, only references to it bound by
//   the source topology's lifetime are exposed.
// - APIs for interacting with topologies and topology objects honor Rust's
//   shared XOR mutable aliasing rules, with no internal mutability.
//
// Provided that objects do not link to other objects outside of the topology
// they originate from, which is minimally sane expectation from hwloc, this
// should be enough.
//
// The hwloc_obj has very complex consistency invariants that are not fully
// documented by upstream. We assume the following:
//
// - If any pointer is non-null, its target can be assumed to be valid
// - Anything that is not explicitly listed as okay to modify below should be
//   considered unsafe to modify unless proven otherwise
// - object_type is assumed to be in sync with attr
// - It is okay to change attr inner data as long as no union is switched
//   from one variant to another
// - subtype may be replaced with another C string allocated by malloc(),
//   which hwloc will automatically free() on topology destruction (source:
//   documentation of hwloc_topology_insert_group_object() encourages it).
//   However, this is only safe if hwloc is linked against the same libc as the
//   application, so it should be made unsafe on Windows where linking against
//   two different CRTs is both easy to do and hard to avoid.
// - depth is in sync with parent
// - logical_index is in sync with (next|prev)_cousin
// - sibling_rank is in sync with (next|prev)_sibling
// - arity is in sync with (children|(first|last)_child)
// - symmetric_subtree is in sync with child pointers
// - memory_arity is in sync with memory_first_child
// - io_arity is in sync with io_first_child
// - misc_arity is in sync with misc_first_child
// - infos_count is in sync with infos
// - userdata should not be touched as topology duplication aliases it
// - gp_index is stable by API contract
#[allow(clippy::non_send_fields_in_send_ty, missing_copy_implementations)]
#[doc(alias = "hwloc_obj")]
#[doc(alias = "hwloc_obj_t")]
#[repr(transparent)]
pub struct TopologyObject(hwloc_obj);

/// # Basic identity
impl TopologyObject {
    /// Type of object
    #[doc(alias = "hwloc_obj::type")]
    pub fn object_type(&self) -> ObjectType {
        self.0.ty.try_into().expect("Got unexpected object type")
    }

    /// Subtype string to better describe the type field
    ///
    /// See <https://hwloc.readthedocs.io/en/v2.9/attributes.html#attributes_normal>
    /// for a list of subtype strings that hwloc can emit.
    #[doc(alias = "hwloc_obj::subtype")]
    pub fn subtype(&self) -> Option<&CStr> {
        // SAFETY: - Pointer validity is assumed as a type invariant
        //         - Rust aliasing rules are enforced by deriving the reference
        //           from &self, which itself is derived from &Topology
        unsafe { ffi::deref_str(&self.0.subtype) }
    }

    /// Set the subtype string
    ///
    /// This exposes [`TopologyObject::set_subtype_unchecked()`] as a safe
    /// method on operating systems which aren't known to facilitate mixing and
    /// matching libc versions between an application and its dependencies.
    #[allow(clippy::missing_errors_doc)]
    #[cfg(all(feature = "hwloc-2_3_0", not(windows)))]
    pub fn set_subtype(&mut self, subtype: &str) -> Result<(), NulError> {
        // SAFETY: Underlying OS is assumed not to ergonomically encourage
        //         unsafe multi-libc linkage
        unsafe { self.set_subtype_unchecked(subtype) }
    }

    /// Set the subtype string
    ///
    /// This is something you'll often want to do when creating Group or Misc
    /// objects in order to make them more descriptive.
    ///
    /// # Safety
    ///
    /// This method is only safe to call if you can guarantee that your
    /// application links against the same libc/CRT as hwloc.
    ///
    /// While linking against a different libc is something that is either
    /// outright UB or strongly advised against on every OS, actually following
    /// this sanity rule is unfortunately very hard on Windows, where usage of
    /// multiple CRTs and pre-built DLLs is the norm, and there is no easy way
    /// to tell which CRT version your pre-built hwloc DLL links against to
    /// adjust your application build's CRT choice accordingly.
    ///
    /// Which is why hwlocality tries hard to accomodate violations of the rule.
    /// And thus the safe version of this method, which assumes that you do
    /// follow the rule, is not available on Windows.
    ///
    /// # Errors
    ///
    /// - [`NulError`] if `subtype` contains NUL chars.
    #[cfg(feature = "hwloc-2_3_0")]
    pub unsafe fn set_subtype_unchecked(&mut self, subtype: &str) -> Result<(), NulError> {
        // SAFETY: Per input precondition
        self.0.subtype = unsafe { LibcString::new(subtype)?.into_raw() };
        Ok(())
    }

    /// Object-specific name, if any
    ///
    /// Mostly used for identifying OS devices and Misc objects where a name
    /// string is more useful than numerical indices.
    #[doc(alias = "hwloc_obj::name")]
    pub fn name(&self) -> Option<&CStr> {
        // SAFETY: - Pointer validity is assumed as a type invariant
        //         - Rust aliasing rules are enforced by deriving the reference
        //           from &self, which itself is derived from &Topology
        unsafe { ffi::deref_str(&self.0.name) }
    }

    /// Object type-specific attributes, if any
    #[doc(alias = "hwloc_obj::attr")]
    pub fn attributes(&self) -> Option<ObjectAttributes<'_>> {
        // SAFETY: Per type invariant
        unsafe { ObjectAttributes::new(self.object_type(), &self.0.attr) }
    }

    /// The OS-provided physical index number
    ///
    /// It is not guaranteed unique across the entire machine,
    /// except for PUs and NUMA nodes.
    ///
    /// Not specified if unknown or irrelevant for this object.
    #[doc(alias = "hwloc_obj::os_index")]
    pub fn os_index(&self) -> Option<usize> {
        (self.0.os_index != HWLOC_UNKNOWN_INDEX).then(|| int::expect_usize(self.0.os_index))
    }

    /// Global persistent index
    ///
    /// Generated by hwloc, unique across the topology (contrary to
    /// [`os_index()`]) and persistent across topology changes (contrary to
    /// [`logical_index()`]).
    ///
    /// All this means you can safely use this index as a cheap key representing
    /// the object in a Set or a Map, as long as that Set or Map only refers to
    /// [`TopologyObject`]s originating from a single [`Topology`].
    ///
    /// [`logical_index()`]: Self::logical_index()
    /// [`os_index()`]: Self::os_index()
    #[doc(alias = "hwloc_obj::gp_index")]
    pub fn global_persistent_index(&self) -> TopologyObjectID {
        self.0.gp_index
    }
}

/// Global persistent [`TopologyObject`] ID
///
/// Generated by hwloc, unique across a given topology and persistent across
/// topology changes. Basically, the only collisions you can expect are between
/// objects from different topologies, which you normally shouldn't mix.
pub type TopologyObjectID = u64;

/// # Depth and ancestors
//
// --- Implementation details ---
//
// Includes functionality inspired by https://hwloc.readthedocs.io/en/v2.9/group__hwlocality__helper__ancestors.html
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
    #[doc(alias = "hwloc_obj::depth")]
    pub fn depth(&self) -> Depth {
        Depth::from_raw(self.0.depth).expect("Got unexpected depth value")
    }

    /// Parent object
    ///
    /// Only `None` for the root `Machine` object.
    #[doc(alias = "hwloc_obj::parent")]
    pub fn parent(&self) -> Option<&Self> {
        // SAFETY: - Pointer & target validity are assumed as a type invariant
        //         - Rust aliasing rules are enforced by deriving the reference
        //           from &self, which itself is derived from &Topology
        unsafe { ffi::deref_ptr_mut(&self.0.parent).map(|raw| raw.as_newtype()) }
    }

    /// Chain of parent objects up to the topology root
    pub fn ancestors(&self) -> impl FusedIterator<Item = &Self> + Clone {
        Ancestors(self)
    }

    /// Search for an ancestor at a certain depth
    ///
    /// `depth` can be a [`Depth`], a [`NormalDepth`] or an [`usize`].
    ///
    /// Will return `None` if the requested depth is deeper than the depth of
    /// the current object.
    #[doc(alias = "hwloc_get_ancestor_obj_by_depth")]
    pub fn ancestor_at_depth<DepthLike>(&self, depth: DepthLike) -> Option<&Self>
    where
        DepthLike: TryInto<Depth>,
        <DepthLike as TryInto<Depth>>::Error: Debug,
    {
        /// Polymorphized version of this function (avoids generics code bloat)
        fn polymorphized(self_: &TopologyObject, depth: Depth) -> Option<&TopologyObject> {
            // Fast failure path when depth is comparable
            let self_depth = self_.depth();
            if let (Ok(self_depth), Ok(depth)) = (
                NormalDepth::try_from(self_depth),
                NormalDepth::try_from(depth),
            ) {
                if self_depth <= depth {
                    return None;
                }
            }

            // Otherwise, walk parents looking for the right depth
            self_.ancestors().find(|ancestor| ancestor.depth() == depth)
        }

        // There cannot be any ancestor at a depth below the hwloc-supported max
        let depth = depth.try_into().ok()?;
        polymorphized(self, depth)
    }

    /// Search for the first ancestor with a certain type in ascending order
    ///
    /// If multiple matching ancestors exist (which can happen with [`Group`]
    /// ancestors), the lowest ancestor is returned.
    ///
    /// Will return `None` if the requested type appears deeper than the
    /// current object or doesn't appear in the topology.
    ///
    /// [`Group`]: ObjectType::Group
    #[doc(alias = "hwloc_get_ancestor_obj_by_type")]
    pub fn first_ancestor_with_type(&self, ty: ObjectType) -> Option<&Self> {
        self.ancestors()
            .find(|ancestor| ancestor.object_type() == ty)
    }

    /// Search for the first ancestor that is shared with another object
    ///
    /// The search will always succeed unless...
    /// - One of `self` and `other` is the root [`Machine`](ObjectType::Machine)
    ///   object, which has no ancestors.
    /// - `self` and `other` do not belong to the same topology, and thus have
    ///   no shared ancestor.
    #[doc(alias = "hwloc_get_common_ancestor_obj")]
    pub fn first_common_ancestor(&self, other: &Self) -> Option<&Self> {
        // Handle degenerate case
        if ptr::eq(self, other) {
            return self.parent();
        }

        /// Collect ancestors with virtual depths on both sides
        /// Returns the list of ancestors with virtual depths together with the
        /// first ancestor with a normal depth, if any
        fn collect_virtual_ancestors(
            obj: &TopologyObject,
        ) -> (Vec<&TopologyObject>, Option<&TopologyObject>) {
            let mut ancestors = Vec::new();
            let mut current = obj;
            loop {
                if let Some(parent) = current.parent() {
                    if let Depth::Normal(_) = parent.depth() {
                        return (ancestors, Some(parent));
                    } else {
                        ancestors.push(parent);
                        current = parent;
                    }
                } else {
                    return (ancestors, None);
                }
            }
        }
        let (virtual_ancestors_1, parent1) = collect_virtual_ancestors(self);
        let (virtual_ancestors_2, parent2) = collect_virtual_ancestors(other);

        // Make sure there is no common ancestor at some virtual depth (can't
        // avoid O(N²) alg here as virtual depths cannot be compared and global
        // persistent indices may be redundant across different topologies)
        for ancestor1 in virtual_ancestors_1 {
            for ancestor2 in &virtual_ancestors_2 {
                if ptr::eq(ancestor1, *ancestor2) {
                    return Some(ancestor1);
                }
            }
        }

        // Now that we have virtual depths taken care of, we can enter a fast
        // path for parents with normal depths (if any)
        let mut parent1 = parent1?;
        let mut parent2 = parent2?;
        loop {
            // Walk up ancestors, try to reach the same depth.
            // Only normal depths should be observed all the way through the
            // ancestor chain, since the parent of a normal object is normal.
            let normal_depth = |obj: &Self| {
                NormalDepth::try_from(obj.depth()).expect("Should only observe normal depth here")
            };
            let depth2 = normal_depth(parent2);
            while normal_depth(parent1) > depth2 {
                parent1 = parent1.parent()?;
            }
            let depth1 = normal_depth(parent1);
            while normal_depth(parent2) > depth1 {
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
            if parent1.depth() == parent2.depth() {
                parent1 = parent1.parent()?;
                parent2 = parent2.parent()?;
            }
        }
    }

    /// Truth that this object is in the subtree beginning with ancestor
    /// object `subtree_root`
    ///
    /// This will return `false` if `self` and `subtree_root` do not belong to
    /// the same topology.
    #[doc(alias = "hwloc_obj_is_in_subtree")]
    pub fn is_in_subtree(&self, subtree_root: &Self) -> bool {
        // NOTE: Not reusing the cpuset-based optimization of hwloc as it is
        //       invalid in the presence of objects that do not belong to the
        //       same topology and there is no way to detect whether this is the
        //       case or not without... walking the ancestors ;)
        self.ancestors()
            .any(|ancestor| ptr::eq(ancestor, subtree_root))
    }

    /// Get the first data (or unified) CPU cache shared between this object and
    /// another object, if any.
    ///
    /// Will always return `None` if called on an I/O or Misc object that does
    /// not contain CPUs.
    #[doc(alias = "hwloc_get_shared_cache_covering_obj")]
    pub fn first_shared_cache(&self) -> Option<&Self> {
        let cpuset = self.cpuset()?;
        self.ancestors()
            .skip_while(|ancestor| {
                ancestor
                    .cpuset()
                    .map_or(false, |ancestor_set| ancestor_set == cpuset)
            })
            .find(|ancestor| ancestor.object_type().is_cpu_data_cache())
    }

    /// Get the first non-I/O ancestor object
    ///
    /// Find the smallest non-I/O ancestor object. This object (normal or
    /// memory) may then be used for binding because it has CPU and node sets
    /// and because its locality is the same as this object.
    ///
    /// This operation will fail if and only if the object is the root of the
    /// topology.
    #[doc(alias = "hwloc_get_non_io_ancestor_obj")]
    pub fn first_non_io_ancestor(&self) -> Option<&Self> {
        self.ancestors().find(|obj| obj.cpuset().is_some())
    }
}

/// Iterator over ancestors of a topology object
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
        let depth_res = usize::try_from(self.0.depth());
        (depth_res.unwrap_or(0), depth_res.ok())
    }
}
//
impl FusedIterator for Ancestors<'_> {}

/// # Cousins and siblings
impl TopologyObject {
    /// Horizontal index in the whole list of similar objects, hence guaranteed
    /// unique across the entire machine
    ///
    /// Could be a "cousin rank" since it's the rank within the "cousin" list.
    ///
    /// Note that this index may change when restricting the topology
    /// or when inserting a group.
    #[doc(alias = "hwloc_obj::logical_index")]
    pub fn logical_index(&self) -> usize {
        int::expect_usize(self.0.logical_index)
    }

    /// Next object of same type and depth
    #[doc(alias = "hwloc_obj::next_cousin")]
    pub fn next_cousin(&self) -> Option<&Self> {
        // SAFETY: - Pointer and target validity are assumed as a type invariant
        //         - Rust aliasing rules are enforced by deriving the reference
        //           from &self, which itself is derived from &Topology
        unsafe { ffi::deref_ptr_mut(&self.0.next_cousin).map(|raw| raw.as_newtype()) }
    }

    /// Previous object of same type and depth
    #[doc(alias = "hwloc_obj::prev_cousin")]
    pub fn prev_cousin(&self) -> Option<&Self> {
        // SAFETY: - Pointer and target validity are assumed as a type invariant
        //         - Rust aliasing rules are enforced by deriving the reference
        //           from &self, which itself is derived from &Topology
        unsafe { ffi::deref_ptr_mut(&self.0.prev_cousin).map(|raw| raw.as_newtype()) }
    }

    /// Index in the parent's relevant child list for this object type
    #[doc(alias = "hwloc_obj::sibling_rank")]
    pub fn sibling_rank(&self) -> usize {
        int::expect_usize(self.0.sibling_rank)
    }

    /// Next object below the same parent, in the same child list
    #[doc(alias = "hwloc_obj::next_sibling")]
    pub fn next_sibling(&self) -> Option<&Self> {
        // SAFETY: - Pointer and target validity are assumed as a type invariant
        //         - Rust aliasing rules are enforced by deriving the reference
        //           from &self, which itself is derived from &Topology
        unsafe { ffi::deref_ptr_mut(&self.0.next_sibling).map(|raw| raw.as_newtype()) }
    }

    /// Previous object below the same parent, in the same child list
    #[doc(alias = "hwloc_obj::prev_sibling")]
    pub fn prev_sibling(&self) -> Option<&Self> {
        // SAFETY: - Pointer and target validity are assumed as a type invariant
        //         - Rust aliasing rules are enforced by deriving the reference
        //           from &self, which itself is derived from &Topology
        unsafe { ffi::deref_ptr_mut(&self.0.prev_sibling).map(|raw| raw.as_newtype()) }
    }
}

/// # Children
impl TopologyObject {
    /// Number of normal children (excluding Memory, Misc and I/O)
    #[doc(alias = "hwloc_obj::arity")]
    pub fn normal_arity(&self) -> usize {
        int::expect_usize(self.0.arity)
    }

    /// Normal children of this object
    #[doc(alias = "hwloc_obj::children")]
    #[doc(alias = "hwloc_obj::first_child")]
    #[doc(alias = "hwloc_obj::last_child")]
    pub fn normal_children(
        &self,
    ) -> impl DoubleEndedIterator<Item = &Self> + Clone + ExactSizeIterator + FusedIterator {
        if self.0.children.is_null() {
            assert_eq!(
                self.normal_arity(),
                0,
                "Got null children pointer with nonzero arity"
            );
        }
        (0..self.normal_arity()).map(move |offset| {
            // SAFETY: Pointer is in bounds by construction
            let child = unsafe { *self.0.children.add(offset) };
            assert!(!child.is_null(), "Got null child pointer");
            // SAFETY: - We checked that the pointer isn't null
            //         - Pointer & target validity assumed as a type invariant
            //         - Rust aliasing rules are enforced by deriving the reference
            //           from &self, which itself is derived from &Topology
            unsafe { (&*child).as_newtype() }
        })
    }

    // NOTE: Not exposing first_/last_child accessors for now as in the presence
    //       of the normal_children iterator, they feel very redundant, and I
    //       can't think of a usage situation where avoiding one pointer
    //       indirection by exposing them would be worth the API inconsistency.
    //       If you do, please submit an issue to the repository!

    /// Truth that this object is symmetric, which means all normal children and
    /// their children have identically shaped subtrees
    ///
    /// Memory, I/O and Misc children are ignored when evaluating this property,
    /// and it is false for all of these object types.
    ///
    /// If this is true of the root object, then the topology may be [exported
    /// as a synthetic string](Topology::export_synthetic()).
    #[doc(alias = "hwloc_obj::symmetric_subtree")]
    pub fn is_symmetric_subtree(&self) -> bool {
        self.0.symmetric_subtree != 0
    }

    /// Get the child covering at least the given cpuset `set`
    ///
    /// `set` can be a `&'_ CpuSet` or a `BitmapRef<'_, CpuSet>`.
    ///
    /// This function will always return `None` if the given set is empty or
    /// this topology object doesn't have a cpuset (I/O or Misc objects), as
    /// no object is considered to cover the empty cpuset.
    #[doc(alias = "hwloc_get_child_covering_cpuset")]
    pub fn normal_child_covering_cpuset(&self, set: impl Deref<Target = CpuSet>) -> Option<&Self> {
        let set: &CpuSet = &set;
        self.normal_children()
            .find(|child| child.covers_cpuset(set))
    }

    /// Number of memory children
    #[doc(alias = "hwloc_obj::memory_arity")]
    pub fn memory_arity(&self) -> usize {
        int::expect_usize(self.0.memory_arity)
    }

    /// Memory children of this object
    ///
    /// NUMA nodes and Memory-side caches are listed here instead of in the
    /// [`TopologyObject::normal_children()`] list. See also
    /// [`ObjectType::is_memory()`].
    ///
    /// A memory hierarchy starts from a normal CPU-side object (e.g.
    /// [`Package`]) and ends with NUMA nodes as leaves. There might exist some
    /// memory-side caches between them in the middle of the memory subtree.
    ///
    /// [`Package`]: ObjectType::Package
    #[doc(alias = "hwloc_obj::memory_first_child")]
    pub fn memory_children(&self) -> impl ExactSizeIterator<Item = &Self> + Clone + FusedIterator {
        // SAFETY: - memory_first_child is a valid first-child of this object
        //         - memory_arity is assumed in sync as a type invariant
        unsafe { self.singly_linked_children(self.0.memory_first_child, self.memory_arity()) }
    }

    /// Total memory (in bytes) in NUMA nodes below this object
    ///
    /// Requires [`DiscoverySupport::numa_memory()`].
    #[doc(alias = "hwloc_obj::total_memory")]
    pub fn total_memory(&self) -> u64 {
        self.0.total_memory
    }

    /// Number of I/O children
    #[doc(alias = "hwloc_obj::io_arity")]
    pub fn io_arity(&self) -> usize {
        int::expect_usize(self.0.io_arity)
    }

    /// I/O children of this object
    ///
    /// Bridges, PCI and OS devices are listed here instead of in the
    /// [`TopologyObject::normal_children()`] list. See also
    /// [`ObjectType::is_io()`].
    #[doc(alias = "hwloc_obj::io_first_child")]
    pub fn io_children(&self) -> impl ExactSizeIterator<Item = &Self> + Clone + FusedIterator {
        // SAFETY: - io_first_child is a valid first-child of this object
        //         - io_arity is assumed in sync as a type invariant
        unsafe { self.singly_linked_children(self.0.io_first_child, self.io_arity()) }
    }

    /// Truth that this is a bridge covering the specified PCI bus
    #[doc(alias = "hwloc_bridge_covers_pcibus")]
    pub fn is_bridge_covering_pci_bus(&self, domain: PCIDomain, bus_id: u8) -> bool {
        let Some(ObjectAttributes::Bridge(bridge)) = self.attributes() else {
            return false;
        };
        let Some(DownstreamAttributes::PCI(pci)) = bridge.downstream_attributes() else {
            // Cannot happen on current hwloc, but may happen someday
            return false;
        };
        pci.domain() == domain && pci.secondary_bus() <= bus_id && pci.subordinate_bus() >= bus_id
    }

    /// Number of Misc children
    #[doc(alias = "hwloc_obj::misc_arity")]
    pub fn misc_arity(&self) -> usize {
        int::expect_usize(self.0.misc_arity)
    }

    /// Misc children of this object
    ///
    /// Misc objects are listed here instead of in the
    /// [`TopologyObject::normal_children()`] list.
    #[doc(alias = "hwloc_obj::misc_first_child")]
    pub fn misc_children(&self) -> impl ExactSizeIterator<Item = &Self> + Clone + FusedIterator {
        // SAFETY: - misc_first_child is a valid first-child of this object
        //         - misc_arity is assumed in sync as a type invariant
        unsafe { self.singly_linked_children(self.0.misc_first_child, self.misc_arity()) }
    }

    /// Full list of children (normal, then memory, then I/O, then Misc)
    #[doc(alias = "hwloc_get_next_child")]
    pub fn all_children(&self) -> impl FusedIterator<Item = &Self> + Clone {
        self.normal_children()
            .chain(self.memory_children())
            .chain(self.io_children())
            .chain(self.misc_children())
    }

    /// Iterator over singly linked lists of child objects with known arity
    ///
    /// # Safety
    ///
    /// - `first` must be one of the `xyz_first_child` pointers of this object
    /// - `arity` must be the matching `xyz_arity` child count variable
    unsafe fn singly_linked_children(
        &self,
        first: *mut hwloc_obj,
        arity: usize,
    ) -> impl ExactSizeIterator<Item = &Self> + Clone + FusedIterator {
        let mut current = first;
        (0..arity).map(move |_| {
            assert!(!current.is_null(), "Got null child before expected arity");
            // SAFETY: - We checked that the pointer isn't null
            //         - Pointer & target validity assumed as a type invariant
            //         - Rust aliasing rules are enforced by deriving the reference
            //           from &self, which itself is derived from &Topology
            let result: &Self = unsafe { (&*current).as_newtype() };
            current = result.0.next_sibling;
            result
        })
    }
}

/// # CPU set
impl TopologyObject {
    /// CPUs covered by this object
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
    /// All objects have CPU and node sets except Misc and I/O objects, so if
    /// you know this object to be a normal or Memory object, you can safely
    /// unwrap this Option.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use hwlocality::Topology;
    /// # let topology = Topology::test_instance();
    /// println!(
    ///     "Visible CPUs attached to the root object: {:?}",
    ///     topology.root_object().cpuset()
    /// );
    /// # Ok::<_, eyre::Report>(())
    /// ```
    #[doc(alias = "hwloc_obj::cpuset")]
    pub fn cpuset(&self) -> Option<BitmapRef<'_, CpuSet>> {
        // SAFETY: Per type invariant
        unsafe { CpuSet::borrow_from_raw_mut(self.0.cpuset) }
    }

    /// Truth that this object is inside of the given cpuset `set`
    ///
    /// `set` can be a `&'_ CpuSet` or a `BitmapRef<'_, CpuSet>`.
    ///
    /// Objects are considered to be inside `set` if they have a non-empty
    /// cpuset which verifies `set.includes(object_cpuset)`.
    pub fn is_inside_cpuset(&self, set: impl Deref<Target = CpuSet>) -> bool {
        let Some(object_cpuset) = self.cpuset() else {
            return false;
        };
        set.includes(object_cpuset) && !object_cpuset.is_empty()
    }

    /// Truth that this object covers the given cpuset `set`
    ///
    /// `set` can be a `&'_ CpuSet` or a `BitmapRef<'_, CpuSet>`.
    ///
    /// Objects are considered to cover `set` if it is non-empty and the object
    /// has a cpuset which verifies `object_cpuset.includes(set)`.
    pub fn covers_cpuset(&self, set: impl Deref<Target = CpuSet>) -> bool {
        let Some(object_cpuset) = self.cpuset() else {
            return false;
        };
        let set: &CpuSet = &set;
        object_cpuset.includes(set) && !set.is_empty()
    }

    /// The complete CPU set of this object
    ///
    /// To the CPUs listed by [`cpuset()`], this adds CPUs for which topology
    /// information is unknown or incomplete, some offline CPUs, and CPUs that
    /// are ignored when the [`BuildFlags::INCLUDE_DISALLOWED`] topology
    /// building configuration flag is not set.
    ///
    /// Thus no corresponding PU object may be found in the topology, because
    /// the precise position is undefined. It is however known that it would be
    /// somewhere under this object.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use hwlocality::Topology;
    /// # let topology = Topology::test_instance();
    /// println!(
    ///     "Overall CPUs attached to the root object: {:?}",
    ///     topology.root_object().complete_cpuset()
    /// );
    /// # Ok::<_, eyre::Report>(())
    /// ```
    ///
    /// [`cpuset()`]: Self::cpuset()
    #[doc(alias = "hwloc_obj::complete_cpuset")]
    pub fn complete_cpuset(&self) -> Option<BitmapRef<'_, CpuSet>> {
        // SAFETY: Per type invariant
        unsafe { CpuSet::borrow_from_raw_mut(self.0.complete_cpuset) }
    }
}

/// # NUMA node set
impl TopologyObject {
    /// NUMA nodes covered by this object or containing this object.
    ///
    /// This is the set of NUMA nodes for which there are NUMA node objects in
    /// the topology under or above this object, i.e. which are known to be
    /// physically contained in this object or containing it and known how
    /// (the children path between this object and the NUMA node objects). In
    /// the end, these nodes are those that are close to the current object.
    ///
    #[cfg_attr(
        feature = "hwloc-2_3_0",
        doc = "With hwloc 2.3+, [`Topology::local_numa_nodes()`] may be used to"
    )]
    #[cfg_attr(feature = "hwloc-2_3_0", doc = "list those NUMA nodes more precisely.")]
    ///
    /// If the [`BuildFlags::INCLUDE_DISALLOWED`] topology building
    /// configuration flag is set, some of these nodes may not be allowed for
    /// allocation, see [`Topology::allowed_nodeset()`].
    ///
    /// If there are no NUMA nodes in the machine, all the memory is close to
    /// this object, so the nodeset is full.
    ///
    /// All objects have CPU and node sets except Misc and I/O objects, so if
    /// you know this object to be a normal or Memory object, you can safely
    /// unwrap this Option.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use hwlocality::Topology;
    /// # let topology = Topology::test_instance();
    /// println!(
    ///     "Visible NUMA nodes attached to the root object: {:?}",
    ///     topology.root_object().nodeset()
    /// );
    /// # Ok::<_, eyre::Report>(())
    /// ```
    #[doc(alias = "hwloc_obj::nodeset")]
    pub fn nodeset(&self) -> Option<BitmapRef<'_, NodeSet>> {
        // SAFETY: Per type invariant
        unsafe { NodeSet::borrow_from_raw_mut(self.0.nodeset) }
    }

    /// The complete NUMA node set of this object
    ///
    /// To the nodes listed by [`nodeset()`], this adds nodes for which topology
    /// information is unknown or incomplete, some offline nodes, and nodes
    /// that are ignored when the [`BuildFlags::INCLUDE_DISALLOWED`] topology
    /// building configuration flag is not set.
    ///
    /// Thus no corresponding [`NUMANode`] object may be found in the topology,
    /// because the precise position is undefined. It is however known that it
    /// would be somewhere under this object.
    ///
    /// If there are no NUMA nodes in the machine, all the memory is close to
    /// this object, so the complete nodeset is full.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use hwlocality::Topology;
    /// # let topology = Topology::test_instance();
    /// println!(
    ///     "Overall NUMA nodes attached to the root object: {:?}",
    ///     topology.root_object().complete_nodeset()
    /// );
    /// # Ok::<_, eyre::Report>(())
    /// ```
    ///
    /// [`nodeset()`]: Self::nodeset()
    /// [`NUMANode`]: ObjectType::NUMANode
    #[doc(alias = "hwloc_obj::complete_nodeset")]
    pub fn complete_nodeset(&self) -> Option<BitmapRef<'_, NodeSet>> {
        // SAFETY: Per type invariant
        unsafe { NodeSet::borrow_from_raw_mut(self.0.complete_nodeset) }
    }
}

/// # Key-value information
impl TopologyObject {
    /// Complete list of (key, value) textual info pairs
    ///
    /// hwloc defines [a number of standard object info attribute names with
    /// associated semantics](https://hwloc.readthedocs.io/en/v2.9/attributes.html#attributes_info).
    ///
    /// Beware that hwloc allows multiple informations with the same key to
    /// exist, although sane users should not leverage this possibility.
    #[doc(alias = "hwloc_obj::infos")]
    pub fn infos(&self) -> &[TextualInfo] {
        // Handle null infos pointer
        if self.0.infos.is_null() {
            assert_eq!(
                self.0.infos_count, 0,
                "Got null infos pointer with nonzero info count"
            );
            return &[];
        }

        // Handle unsupported size slice edge case
        let infos_len = int::expect_usize(self.0.infos_count);
        #[allow(clippy::missing_docs_in_private_items)]
        type Element = TextualInfo;
        int::assert_slice_len::<Element>(infos_len);

        // Build the output slice
        // SAFETY: - infos and count are assumed in sync per type invariant
        //         - infos are assumed to be valid per type invariant
        //         - AsNewtype is trusted to be implemented correctly
        //         - infos_len was checked to be slice-compatible above
        unsafe { std::slice::from_raw_parts::<Element>(self.0.infos.as_newtype(), infos_len) }
    }

    /// Search the given key name in object infos and return the corresponding value
    ///
    /// Beware that hwloc allows multiple informations with the same key to
    /// exist, although no sane programs should leverage this possibility.
    /// If multiple keys match the given name, only the first one is returned.
    ///
    /// Calling this operation multiple times will result in duplicate work. If
    /// you need to do this sort of search many times, consider collecting
    /// `infos()` into a `HashMap` or `BTreeMap` for increased lookup efficiency.
    #[doc(alias = "hwloc_obj_get_info_by_name")]
    pub fn info(&self, key: &str) -> Option<&CStr> {
        self.infos().iter().find_map(|info| {
            let Ok(info_name) = info.name().to_str() else {
                // hwloc does not currently emit invalid Unicode, but it might
                // someday if a malicious C program tampered with the topology
                return None;
            };
            (info_name == key).then_some(info.value())
        })
    }

    /// Add the given info name and value pair to the given object
    ///
    /// The info is appended to the existing info array even if another key with
    /// the same name already exists.
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
    #[cfg(feature = "hwloc-2_3_0")]
    #[doc(alias = "hwloc_obj_add_info")]
    pub fn add_info(&mut self, name: &str, value: &str) -> Result<(), HybridError<NulError>> {
        let name = LibcString::new(name)?;
        let value = LibcString::new(value)?;
        // SAFETY: - An &mut TopologyObject may only be obtained from &mut Topology
        //         - Object validity trusted by type invariant
        //         - hwloc is trusted not to make object invalid
        //         - LibcStrings are valid C strings by construction, and not
        //           used after the end of their lifetimes
        errors::call_hwloc_int_normal("hwloc_obj_add_info", || unsafe {
            hwlocality_sys::hwloc_obj_add_info(&mut self.0, name.borrow(), value.borrow())
        })
        .map(std::mem::drop)
        .map_err(HybridError::Hwloc)
    }
}

// # Internal utilities
impl TopologyObject {
    /// Display this object's type and attributes
    fn display(&self, f: &mut fmt::Formatter<'_>, verbose: bool) -> fmt::Result {
        // SAFETY: - These are indeed snprintf-like APIs
        //         - Object validity trusted by type invariant
        //         - verbose translates nicely into a C-style boolean
        //         - separators are valid C strings
        let (type_chars, attr_chars) = unsafe {
            let type_chars = ffi::call_snprintf(|buf, len| {
                hwlocality_sys::hwloc_obj_type_snprintf(buf, len, &self.0, verbose.into())
            });

            let separator = if f.alternate() {
                b",\n  \0".as_ptr()
            } else {
                b", \0".as_ptr()
            }
            .cast::<c_char>();
            let attr_chars = ffi::call_snprintf(|buf, len| {
                hwlocality_sys::hwloc_obj_attr_snprintf(
                    buf,
                    len,
                    &self.0,
                    separator,
                    verbose.into(),
                )
            });
            (type_chars, attr_chars)
        };

        let cpuset_str = self
            .cpuset()
            .map_or_else(String::new, |cpuset| format!(" with {cpuset}"));

        // SAFETY: - Output of call_snprintf should be valid C strings
        //         - We're not touching type_chars and attr_chars while type_str
        //           and attr_str are live.
        unsafe {
            let type_str = CStr::from_ptr(type_chars.as_ptr()).to_string_lossy();
            let attr_str = CStr::from_ptr(attr_chars.as_ptr()).to_string_lossy();
            let type_and_cpuset = format!("{type_str}{cpuset_str}");
            if attr_str.is_empty() {
                f.pad(&type_and_cpuset)
            } else if f.alternate() {
                let s = format!("{type_and_cpuset} (\n  {attr_str}\n)");
                f.pad(&s)
            } else {
                let s = format!("{type_and_cpuset} ({attr_str})");
                f.pad(&s)
            }
        }
    }

    /// Delete all cpusets and nodesets from a non-inserted `Group` object
    ///
    /// This is needed as part of a dirty topology editing workaround that will
    /// hopefully not be needed anymore after hwloc v2.10.
    ///
    /// # (Absence of) Panics
    ///
    /// This method is called inside of destructors, it should never panic.
    ///
    /// # Safety
    ///
    /// `self_` must designate a valid `Group` object that has been allocated
    /// with `hwloc_topology_alloc_group_object()` but not yet inserted into a
    /// topology with `hwloc_topology_insert_group_object()`.
    #[cfg(all(feature = "hwloc-2_3_0", not(feature = "hwloc-2_10_0")))]
    pub(crate) unsafe fn delete_all_sets(self_: ptr::NonNull<Self>) {
        let self_ = self_.as_ptr();
        debug_assert_eq!(
            // SAFETY: self_ is valid per input precondition
            unsafe { (*self_).0.ty },
            hwlocality_sys::HWLOC_OBJ_GROUP,
            "this method should only be called on Group objects"
        );
        // SAFETY: This is safe per the input precondition that `self_` is a
        //         valid `TopologyObject` (which includes valid bitmap
        //         pointers), and it's not part of a `Topology` yet so we
        //         assume complete ownership of it delete its cpu/node-sets
        //         without worrying about unintended consequences.
        unsafe {
            for set_ptr in [
                ptr::addr_of_mut!((*self_).0.cpuset),
                ptr::addr_of_mut!((*self_).0.nodeset),
                ptr::addr_of_mut!((*self_).0.complete_cpuset),
                ptr::addr_of_mut!((*self_).0.complete_nodeset),
            ] {
                let set = set_ptr.read();
                if !set.is_null() {
                    hwlocality_sys::hwloc_bitmap_free(set);
                    set_ptr.write(ptr::null_mut())
                }
            }
        }
    }
}

impl Debug for TopologyObject {
    /// Verbose display of the object's type and attributes
    ///
    /// See the [`Display`] implementation if you want a more concise display.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use hwlocality::Topology;
    /// # let topology = Topology::test_instance();
    /// println!("Root object: {:#?}", topology.root_object());
    /// # Ok::<_, eyre::Report>(())
    /// ```
    #[doc(alias = "hwloc_obj_attr_snprintf")]
    #[doc(alias = "hwloc_obj_type_snprintf")]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.display(f, true)
    }
}

impl Display for TopologyObject {
    #[allow(clippy::doc_markdown)]
    /// Display of the type and attributes that is more concise than [`Debug`]
    ///
    /// - Shorter type names are used, e.g. "L1Cache" becomes "L1"
    /// - Only the major object attributes are printed
    ///
    /// # Example
    ///
    /// ```rust
    /// # use hwlocality::Topology;
    /// # let topology = Topology::test_instance();
    /// println!("Root object: {}", topology.root_object());
    /// # Ok::<_, eyre::Report>(())
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.display(f, false)
    }
}

// SAFETY: No internal mutability
unsafe impl Send for TopologyObject {}

// SAFETY: No internal mutability
unsafe impl Sync for TopologyObject {}

// SAFETY: TopologyObject is a repr(transparent) newtype of hwloc_obj
unsafe impl TransparentNewtype for TopologyObject {
    type Inner = hwloc_obj;
}

#[allow(clippy::cognitive_complexity)]
#[cfg(test)]
pub(crate) mod tests {
    use super::hierarchy::tests::{any_hwloc_depth, any_normal_depth, any_usize_depth};
    use super::*;
    use crate::{
        strategies::{any_object, any_string, set_with_reference, test_object},
        topology::Topology,
    };
    use proptest::prelude::*;
    use similar_asserts::assert_eq;
    use std::{collections::HashMap, ffi::CString, ops::RangeInclusive};

    /// Run [`check_any_object()`] on every topology object
    #[test]
    fn check_all_objects() -> Result<(), TestCaseError> {
        let topology = Topology::test_instance();
        for obj in topology.objects() {
            check_any_object(obj)?;
        }
        Ok(())
    }

    /// Stuff that should be true of any object we examine
    fn check_any_object(obj: &TopologyObject) -> Result<(), TestCaseError> {
        check_sets(obj)?;
        check_parent(obj)?;
        check_first_shared_cache(obj)?;
        check_cousins_and_siblings(obj)?;
        check_children(obj)?;
        check_infos(obj)?;
        check_displays(obj)?;
        Ok(())
    }

    /// Check that an object's cpusets and nodesets have the expected properties
    fn check_sets(obj: &TopologyObject) -> Result<(), TestCaseError> {
        let has_sets = obj.object_type().has_sets();

        prop_assert_eq!(obj.cpuset().is_some(), has_sets);
        prop_assert_eq!(obj.complete_cpuset().is_some(), has_sets);
        if let (Some(complete), Some(normal)) = (obj.complete_cpuset(), obj.cpuset()) {
            prop_assert!(complete.includes(normal));
            prop_assert!(obj.is_inside_cpuset(complete));
            prop_assert!(obj.covers_cpuset(normal));
        }

        prop_assert_eq!(obj.nodeset().is_some(), has_sets);
        prop_assert_eq!(obj.complete_nodeset().is_some(), has_sets);
        if let (Some(complete), Some(normal)) = (obj.complete_nodeset(), obj.nodeset()) {
            prop_assert!(complete.includes(normal));
        }

        Ok(())
    }

    /// Check that an object's parent has the expected properties
    fn check_parent(obj: &TopologyObject) -> Result<(), TestCaseError> {
        let Some(parent) = obj.parent() else {
            prop_assert_eq!(obj.object_type(), ObjectType::Machine);
            return Ok(());
        };

        let first_ancestor = obj.ancestors().next().unwrap();
        prop_assert!(ptr::eq(parent, first_ancestor));

        if let (Depth::Normal(parent_depth), Depth::Normal(obj_depth)) =
            (parent.depth(), obj.depth())
        {
            prop_assert!(parent_depth < obj_depth);
        }

        if obj.object_type().has_sets() {
            prop_assert!(parent.object_type().has_sets());
            prop_assert!(parent.cpuset().unwrap().includes(obj.cpuset().unwrap()));
            prop_assert!(parent.nodeset().unwrap().includes(obj.nodeset().unwrap()));
            prop_assert!(parent
                .complete_cpuset()
                .unwrap()
                .includes(obj.complete_cpuset().unwrap()));
            prop_assert!(parent
                .complete_nodeset()
                .unwrap()
                .includes(obj.complete_nodeset().unwrap()));
        }

        Ok(())
    }

    /// Check that [`TopologyObject::first_shared_cache()`] works as expected
    fn check_first_shared_cache(obj: &TopologyObject) -> Result<(), TestCaseError> {
        // Call the function to begin with
        let result = obj.first_shared_cache();

        // Should not yield result on objects without cpusets
        if obj.cpuset().is_none() {
            prop_assert!(result.is_none());
            return Ok(());
        }

        // Otherwise, should yield first cache parent that has multiple normal
        // children below it
        let expected = obj
            .ancestors()
            .skip_while(|ancestor| ancestor.normal_arity() == 1)
            .find(|ancestor| ancestor.object_type().is_cpu_data_cache());
        if let (Some(result), Some(expected)) = (result, expected) {
            prop_assert!(ptr::eq(result, expected));
        } else {
            prop_assert!(result.is_none() && expected.is_none());
        }
        Ok(())
    }

    /// Check that an object's cousin and siblings have the expected properties
    fn check_cousins_and_siblings(obj: &TopologyObject) -> Result<(), TestCaseError> {
        let siblings_len = if let Some(parent) = obj.parent() {
            let ty = obj.object_type();
            if ty.is_normal() {
                parent.normal_arity()
            } else if ty.is_memory() {
                parent.memory_arity()
            } else if ty.is_io() {
                parent.io_arity()
            } else {
                prop_assert_eq!(ty, ObjectType::Misc);
                parent.misc_arity()
            }
        } else {
            1
        };
        let topology = Topology::test_instance();
        let cousins_len = topology.num_objects_at_depth(obj.depth());

        if let Some(prev_cousin) = obj.prev_cousin() {
            check_cousin(obj, prev_cousin)?;
            prop_assert_eq!(prev_cousin.logical_index(), obj.logical_index() - 1);
            prop_assert!(ptr::eq(prev_cousin.next_cousin().unwrap(), obj));
        } else {
            prop_assert_eq!(obj.logical_index(), 0);
        }
        if let Some(next_cousin) = obj.next_cousin() {
            check_cousin(obj, next_cousin)?;
            prop_assert_eq!(next_cousin.logical_index(), obj.logical_index() + 1);
            prop_assert!(ptr::eq(next_cousin.prev_cousin().unwrap(), obj));
        } else {
            prop_assert_eq!(obj.logical_index(), cousins_len - 1);
        }

        if let Some(prev_sibling) = obj.prev_sibling() {
            check_sibling(obj, prev_sibling)?;
            prop_assert_eq!(prev_sibling.sibling_rank(), obj.sibling_rank() - 1);
            prop_assert!(ptr::eq(prev_sibling.next_sibling().unwrap(), obj));
        } else {
            prop_assert_eq!(obj.sibling_rank(), 0);
        }
        if let Some(next_sibling) = obj.next_sibling() {
            check_sibling(obj, next_sibling)?;
            prop_assert_eq!(next_sibling.sibling_rank(), obj.sibling_rank() + 1);
            prop_assert!(ptr::eq(next_sibling.prev_sibling().unwrap(), obj));
        } else {
            prop_assert_eq!(obj.sibling_rank(), siblings_len - 1);
        }

        Ok(())
    }

    /// Check that an object's cousin has the expected properties
    fn check_cousin(obj: &TopologyObject, cousin: &TopologyObject) -> Result<(), TestCaseError> {
        prop_assert_eq!(cousin.object_type(), obj.object_type());
        prop_assert_eq!(cousin.depth(), obj.depth());
        if obj.object_type().has_sets() {
            prop_assert!(!obj.cpuset().unwrap().intersects(cousin.cpuset().unwrap()));
            prop_assert!(!obj
                .complete_cpuset()
                .unwrap()
                .intersects(cousin.complete_cpuset().unwrap()));
        }
        Ok(())
    }

    /// Check that an object's sibling has the expected properties
    ///
    /// This does not re-check the properties checked by `check_cousin()`, since
    /// the sibling of some object must be the cousin of another object.
    fn check_sibling(obj: &TopologyObject, sibling: &TopologyObject) -> Result<(), TestCaseError> {
        prop_assert_eq!(sibling.0.parent, obj.0.parent);
        Ok(())
    }

    /// Check that an object's children has the expected properties
    fn check_children(obj: &TopologyObject) -> Result<(), TestCaseError> {
        prop_assert_eq!(obj.normal_arity(), obj.normal_children().count());
        prop_assert_eq!(obj.memory_arity(), obj.memory_children().count());
        prop_assert_eq!(obj.io_arity(), obj.io_children().count());
        prop_assert_eq!(obj.misc_arity(), obj.misc_children().count());
        prop_assert_eq!(
            obj.all_children().count(),
            obj.normal_arity() + obj.memory_arity() + obj.io_arity() + obj.misc_arity()
        );

        // NOTE: Most parent-child relations are checked when checking the
        //       parent, since that's agnostic to the kind of child we deal with
        for (idx, normal_child) in obj.normal_children().enumerate() {
            prop_assert_eq!(normal_child.sibling_rank(), idx);
            prop_assert!(normal_child.object_type().is_normal());
        }
        for (idx, memory_child) in obj.memory_children().enumerate() {
            prop_assert_eq!(memory_child.sibling_rank(), idx);
            prop_assert!(memory_child.object_type().is_memory());
        }
        for (idx, io_child) in obj.io_children().enumerate() {
            prop_assert_eq!(io_child.sibling_rank(), idx);
            prop_assert!(io_child.object_type().is_io());
        }
        for (idx, misc_child) in obj.misc_children().enumerate() {
            prop_assert_eq!(misc_child.sibling_rank(), idx);
            prop_assert_eq!(misc_child.object_type(), ObjectType::Misc);
        }

        Ok(())
    }

    /// Check that an object's info metadata matches expectations
    fn check_infos(obj: &TopologyObject) -> Result<(), TestCaseError> {
        for info in obj.infos() {
            if let Ok(name) = info.name().to_str() {
                prop_assert_eq!(
                    obj.info(name),
                    obj.infos()
                        .iter()
                        .find(|other_info| other_info.name() == info.name())
                        .map(TextualInfo::value)
                );
            }
        }
        // NOTE: Looking up invalid info names is tested elsewhere
        Ok(())
    }

    /// Check that an object's displays match expectations
    fn check_displays(obj: &TopologyObject) -> Result<(), TestCaseError> {
        // Render all supported display flavors
        let display = format!("{obj}");
        let display_alternate = format!("{obj:#}");
        let debug = format!("{obj:?}");
        let debug_alternate = format!("{obj:#?}");

        // Non-alternate displays should fit in a single line
        for non_alternate in [&display, &debug] {
            prop_assert!(!non_alternate.contains('\n'));
        }

        // Debug output should be longer than or identical to Display output
        prop_assert!(debug.len() >= display.len());
        prop_assert!(debug_alternate.len() >= display_alternate.len());

        // Alternate displays should be longer than or identical to the norm
        prop_assert!(debug_alternate.len() >= debug.len());
        prop_assert!(display_alternate.len() >= display.len());
        Ok(())
    }

    /// Check that [`TopologyObject::is_symmetric_subtree()`] is correct
    #[test]
    fn is_symmetric_subtree() {
        // Iterate over topology objects from children to parent: by the time we
        // reach a parent, we know that we can trust the is_symmetric_subtree of
        // its direct (and transitive) normal children.
        let topology = Topology::test_instance();
        for depth in NormalDepth::iter_range(NormalDepth::MIN, topology.depth()).rev() {
            'objs: for obj in topology.objects_at_depth(depth) {
                // An object is a symmetric subtree if it has no children...
                let Some(first_child) = obj.normal_children().next() else {
                    assert!(obj.is_symmetric_subtree());
                    continue 'objs;
                };

                // ...or if its children all have the following properties:
                let should_be_symmetric = obj.normal_children().all(|child| {
                    // - They are symmetric subtree themselves
                    child.is_symmetric_subtree()

                    // - All of their topologically meaningful properties are
                    //   identical (and thus equal to that of first child)
                    && child.object_type() == first_child.object_type()
                    && child.subtype() == first_child.subtype()
                    && child.attributes() == first_child.attributes()
                    && child.depth() == first_child.depth()
                    && child.normal_arity() == first_child.normal_arity()
                });
                assert_eq!(obj.is_symmetric_subtree(), should_be_symmetric);
            }
        }

        // Only normal objects can be symmetric
        assert!(topology
            .virtual_objects()
            .all(|obj| !obj.is_symmetric_subtree()));
    }

    /// Check that [`TopologyObject::total_memory()`] is correct
    #[test]
    fn total_memory() {
        // We'll compute the expected total amount of memory below each NUMA
        // node through a bottom-up tree reduction from NUMA nodes up
        let topology = Topology::test_instance();
        let mut expected_total_memory = HashMap::new();
        let mut curr_objects = HashMap::new();
        let mut next_objects = HashMap::new();

        // Seed tree reduction by counting local memory inside of each NUMA node
        for numa in topology.objects_with_type(ObjectType::NUMANode) {
            let Some(ObjectAttributes::NUMANode(attrs)) = numa.attributes() else {
                unreachable!()
            };
            let gp_index = numa.global_persistent_index();
            let local_memory = attrs.local_memory().map_or(0, u64::from);
            assert!(expected_total_memory
                .insert(gp_index, local_memory)
                .is_none());
            assert!(curr_objects.insert(gp_index, numa).is_none());
        }

        // Compute expected total_memory value through tree reduction
        while !curr_objects.is_empty() {
            for (gp_index, obj) in curr_objects.drain() {
                let obj_memory = expected_total_memory[&gp_index];
                if let Some(parent) = obj.parent() {
                    let parent_gp_index = parent.global_persistent_index();
                    *expected_total_memory.entry(parent_gp_index).or_default() += obj_memory;
                    next_objects.insert(parent_gp_index, parent);
                }
            }
            std::mem::swap(&mut curr_objects, &mut next_objects);
        }

        // At this point we should have built an accurate map of how much memory
        // lies in NUMA nodes below each object, use it to check every object.
        for obj in topology.objects() {
            assert_eq!(
                obj.total_memory(),
                expected_total_memory
                    .remove(&obj.global_persistent_index())
                    .unwrap_or(0)
            );
        }
    }

    // --- Test operations with a depth parameter ---

    /// Test [`TopologyObject::ancestor_at_depth()`] for a certain
    /// [`TopologyObject`] and at a certain desired parent depth
    fn check_ancestor_at_depth<DepthLike>(
        obj: &TopologyObject,
        depth: DepthLike,
    ) -> Result<(), TestCaseError>
    where
        DepthLike: TryInto<Depth> + Copy + Debug + Eq,
        Depth: PartialEq<DepthLike>,
        <DepthLike as TryInto<Depth>>::Error: Debug,
    {
        let actual = obj.ancestor_at_depth(depth);
        let expected = obj.ancestors().find(|obj| obj.depth() == depth);
        if let (Some(actual), Some(expected)) = (actual, expected) {
            prop_assert!(ptr::eq(actual, expected));
        } else {
            prop_assert!(actual.is_none() && expected.is_none());
        }
        Ok(())
    }

    proptest! {
        // Probe ancestors by depth at valid and invalid depths
        #[test]
        fn ancestor_at_hwloc_depth(obj in test_object(),
                                   depth in any_hwloc_depth()) {
            check_ancestor_at_depth(obj, depth)?;
        }
        //
        #[test]
        fn ancestor_at_normal_depth(obj in test_object(),
                                    depth in any_normal_depth()) {
            check_ancestor_at_depth(obj, depth)?;
        }
        //
        #[test]
        fn ancestor_at_usize_depth(obj in test_object(),
                                   depth in any_usize_depth()) {
            check_ancestor_at_depth(obj, depth)?;
        }
    }

    // --- Querying stuff by cpuset/nodeset ---

    /// Pick an object and a related cpuset
    pub(crate) fn object_and_related_cpuset(
    ) -> impl Strategy<Value = (&'static TopologyObject, CpuSet)> {
        // Separate objects with and without cpusets
        let topology = Topology::test_instance();
        let mut with_cpuset = Vec::new();
        let mut without_cpuset = Vec::new();
        for obj in topology.objects() {
            if obj.object_type().has_sets() {
                with_cpuset.push(obj);
            } else {
                without_cpuset.push(obj);
            }
        }

        // For objects with cpusets, the reference cpuset is their cpuset
        fn with_reference(
            objects: Vec<&'static TopologyObject>,
            ref_cpuset: impl Fn(&TopologyObject) -> BitmapRef<'_, CpuSet>,
        ) -> Option<impl Strategy<Value = (&'static TopologyObject, CpuSet)>> {
            (!objects.is_empty()).then(move || {
                prop::sample::select(objects)
                    .prop_flat_map(move |obj| (Just(obj), set_with_reference(ref_cpuset(obj))))
            })
        }
        let with_cpuset = with_reference(with_cpuset, |obj| obj.cpuset().unwrap());

        // For objects without cpusets, the reference is the topology cpuset
        let without_cpuset = with_reference(without_cpuset, |_obj| topology.cpuset());

        // We pick from either list with equal probability
        match (with_cpuset, without_cpuset) {
            (Some(with), Some(without)) => prop_oneof![with, without].boxed(),
            (Some(with), None) => with.boxed(),
            (None, Some(without)) => without.boxed(),
            (None, None) => unreachable!(),
        }
    }

    proptest! {
        /// Test [`TopologyObject::normal_child_covering_cpuset()`]
        #[test]
        fn normal_child_covering_cpuset((obj, set) in object_and_related_cpuset()) {
            if let Some(result) = obj.normal_child_covering_cpuset(&set) {
                prop_assert!(result.covers_cpuset(&set));
            } else {
                prop_assert!(obj.normal_children().all(|obj| !obj.covers_cpuset(&set)));
            }
        }

        /// Test [`TopologyObject::is_inside_cpuset()`]
        #[test]
        fn is_inside_cpuset((obj, set) in object_and_related_cpuset()) {
            let result = obj.is_inside_cpuset(&set);
            let Some(obj_set) = obj.cpuset() else {
                prop_assert!(!result);
                return Ok(());
            };
            if obj_set.is_empty() {
                prop_assert!(!result);
                return Ok(());
            }
            prop_assert_eq!(result, set.includes(obj_set));
        }

        /// Test [`TopologyObject::covers_cpuset()`]
        #[test]
        fn covers_cpuset((obj, set) in object_and_related_cpuset()) {
            let result = obj.covers_cpuset(&set);
            let Some(obj_set) = obj.cpuset() else {
                prop_assert!(!result);
                return Ok(());
            };
            if set.is_empty() {
                prop_assert!(!result);
                return Ok(());
            }
            prop_assert_eq!(result, obj_set.includes(&set));
        }
    }

    // --- Truth that an object is a bridge covering a certain PCI bus ---

    /// Generate queries that have a reasonable chance of returning `true`
    fn bridge_coverage() -> impl Strategy<Value = (&'static TopologyObject, PCIDomain, u8)> {
        #[derive(Clone, Debug)]
        struct BridgeCoverage {
            bridge: &'static TopologyObject,
            domain: PCIDomain,
            bus_id_range: RangeInclusive<u8>,
        }
        let topology = Topology::test_instance();
        let bridge_coverages = topology
            .objects_with_type(ObjectType::Bridge)
            .filter_map(|bridge| {
                let Some(ObjectAttributes::Bridge(attrs)) = bridge.attributes() else {
                    unreachable!()
                };
                let Some(DownstreamAttributes::PCI(pci)) = attrs.downstream_attributes() else {
                    return None;
                };
                Some(BridgeCoverage {
                    bridge,
                    domain: pci.domain(),
                    bus_id_range: pci.secondary_bus()..=pci.subordinate_bus(),
                })
            })
            .collect::<Vec<_>>();
        if bridge_coverages.is_empty() {
            (any_object(), any::<PCIDomain>(), any::<u8>()).boxed()
        } else {
            prop::sample::select(bridge_coverages)
                .prop_flat_map(|bridge_coverage| {
                    let obj = prop_oneof![
                        3 => Just(bridge_coverage.bridge),
                        2 => any_object()
                    ];
                    let domain = prop_oneof![
                        3 => Just(bridge_coverage.domain),
                        2 => any::<PCIDomain>()
                    ];
                    let bus_id = prop_oneof![
                        3 => bridge_coverage.bus_id_range,
                        2 => any::<u8>()
                    ];
                    (obj, domain, bus_id)
                })
                .boxed()
        }
    }

    proptest! {
        /// Test [`TopologyObject::is_bridge_covering_pci_bus()`]
        #[test]
        fn is_bridge_covering_pci_bus((obj, domain, bus_id) in bridge_coverage()) {
            let result = obj.is_bridge_covering_pci_bus(domain, bus_id);
            let Some(ObjectAttributes::Bridge(attrs)) = obj.attributes() else {
                prop_assert!(!result);
                return Ok(());
            };
            let Some(DownstreamAttributes::PCI(pci)) = attrs.downstream_attributes() else {
                prop_assert!(!result);
                return Ok(());
            };
            prop_assert_eq!(
                result,
                domain == pci.domain() && bus_id >= pci.secondary_bus() && bus_id <= pci.subordinate_bus()
            );
        }
    }

    // --- Looking up textual object info by random string ---

    proptest! {
        /// Test [`TopologyObject::info()`] against a random key
        ///
        /// Correct keys are checked in another test.
        #[test]
        fn info(obj in any_object(), key in any_string()) {
            let result = obj.info(&key);
            let Ok(ckey) = CString::new(key.clone()) else {
                assert_eq!(result, None);
                return Ok(());
            };
            assert_eq!(
                obj.info(&key),
                obj.infos().iter().find_map(|info|{
                    (info.name() == ckey.as_c_str()).then_some(info.value())
                })
            );
        }
    }

    // --- Properties of pairs of objects ---

    proptest! {
        /// Test for [`TopologyObject::first_common_ancestor()`]
        #[test]
        fn first_common_ancestor(obj in test_object(), other in any_object()) {
            // Check result
            let result = obj.first_common_ancestor(other);

            // The topology root has no ancestor
            if obj.object_type() == ObjectType::Machine || other.object_type() == ObjectType::Machine
            {
                prop_assert!(result.is_none());
                return Ok(());
            }

            // Objects from two different topologies have no common ancestor
            let topology = Topology::test_instance();
            if !topology.contains(other) {
                prop_assert!(result.is_none());
                return Ok(());
            }

            // Otherwise, there should be a common ancestor
            let common_ancestor = result.unwrap();

            // Check if it is indeed the first common ancestor
            fn prev_ancestor_candidate<'obj>(
                obj: &'obj TopologyObject,
                common_ancestor: &TopologyObject,
            ) -> Option<&'obj TopologyObject> {
                obj.ancestors().take_while(|&ancestor| !ptr::eq(ancestor, common_ancestor)).last()
            }
            let obj_ancestor = prev_ancestor_candidate(obj, common_ancestor);
            let other_ancestor = prev_ancestor_candidate(other, common_ancestor);
            if let (Some(obj_ancestor), Some(other_ancestor)) = (obj_ancestor, other_ancestor) {
                prop_assert!(!ptr::eq(obj_ancestor, other_ancestor));
            }
        }
    }

    // --- Object editing ---

    #[cfg(feature = "hwloc-2_3_0")]
    mod editing {
        use super::*;
        use std::panic::UnwindSafe;

        // Check that a certain object editing method works
        fn test_object_editing<R>(check: impl FnOnce(&mut TopologyObject) -> R + UnwindSafe) -> R {
            let mut topology = Topology::test_instance().clone();
            topology.edit(|editor| {
                let misc = editor
                    .insert_misc_object("This is a modifiable test object trololol", |topology| {
                        topology.root_object()
                    })
                    .unwrap();
                check(misc)
            })
        }

        proptest! {
            // Try to set an object's subtype
            #[cfg(not(windows))]
            #[test]
            fn set_subtype(subtype in any_string()) {
                test_object_editing(|obj| {
                    // Try to set an object's subtype
                    let res = obj.set_subtype(&subtype);

                    // Handle inner NULs
                    if subtype.contains('\0') {
                        prop_assert_eq!(res, Err(NulError));
                        return Ok(());
                    }

                    // Assume success otherwise
                    res.unwrap();
                    prop_assert_eq!(obj.subtype().unwrap().to_str().unwrap(), subtype);
                    Ok(())
                })?;
            }

            // Try to add an info (key, value) pair to an object
            #[test]
            fn add_info(name in any_string(), value in any_string()) {
                test_object_editing(|obj| {
                    // Try to add a (key, value) pair
                    let res = obj.add_info(&name, &value);

                    // Handle inner NULs
                    if name.contains('\0') || value.contains('\0') {
                        prop_assert_eq!(res, Err(NulError.into()));
                        return Ok(());
                    }

                    // Assume success otherwise
                    res.unwrap();
                    prop_assert_eq!(obj.info(&name).unwrap().to_str().unwrap(), value);
                    Ok(())
                })?;
            }
        }
    }
}
