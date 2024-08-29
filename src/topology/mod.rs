//! Hardware topology (main hwloc API entry point)
//!
//! A [`Topology`] contains everything hwloc knows about the hardware and
//! software structure of a system. Among other things, it can be used to query
//! the system topology and to bind threads and processes to hardware CPU cores
//! and NUMA nodes. It is the main entry point of the hwloc API through which
//! almost any other feature of the library is accessed.

pub mod builder;
#[cfg(feature = "hwloc-2_3_0")]
pub mod editor;
pub mod export;
pub mod support;

use self::{
    builder::{BuildFlags, TopologyBuilder, TypeFilter},
    support::FeatureSupport,
};
#[cfg(all(feature = "hwloc-2_3_0", doc))]
use crate::topology::support::MiscSupport;
use crate::{
    bitmap::{Bitmap, BitmapRef, SpecializedBitmap},
    cpu::cpuset::CpuSet,
    errors::{self, ForeignObjectError, RawHwlocError},
    ffi::transparent::AsNewtype,
    memory::nodeset::NodeSet,
    object::{
        depth::{Depth, NormalDepth},
        types::ObjectType,
        TopologyObject,
    },
};
use bitflags::bitflags;
use errno::Errno;
use hwlocality_sys::{
    hwloc_bitmap_s, hwloc_distrib_flags_e, hwloc_topology, hwloc_type_filter_e,
    HWLOC_DISTRIB_FLAG_REVERSE,
};
use libc::EINVAL;
#[allow(unused)]
#[cfg(test)]
use similar_asserts::assert_eq;
use std::{
    collections::BTreeMap,
    fmt::{self, Debug, Pointer},
    ops::Deref,
    ptr::{self, NonNull},
    sync::OnceLock,
};
use thiserror::Error;

/// Main entry point to the hwloc API
///
/// A `Topology` contains everything hwloc knows about the hardware and software
/// structure of a system. It can be used to query the system topology and to
/// bind threads and processes to hardware CPU cores and NUMA nodes.
///
/// Since there are **many** things you can do with a `Topology`, the API is
/// broken down into sections roughly following the structure of the upstream
/// hwloc documentation:
///
/// - [Topology building](#topology-building)
/// - [Full object lists](#full-object-lists) (specific to Rust bindings)
/// - [Object levels, depths and types](#object-levels-depths-and-types)
/// - [CPU cache statistics](#cpu-cache-statistics) (specific to Rust bindings)
/// - [CPU binding](#cpu-binding)
/// - [Memory binding](#memory-binding)
/// - [Modifying a loaded topology](#modifying-a-loaded-topology)
/// - [Finding objects inside a CPU set](#finding-objects-inside-a-cpu-set)
/// - [Finding objects covering at least a CPU set](#finding-objects-covering-at-least-a-cpu-set)
/// - [Finding other objects](#finding-other-objects)
/// - [Distributing work items over a topology](#distributing-work-items-over-a-topology)
/// - [CPU and node sets of entire topologies](#cpu-and-node-sets-of-entire-topologies)
/// - [Finding I/O objects](#finding-io-objects)
/// - [Exporting Topologies to XML](#exporting-topologies-to-xml)
/// - [Exporting Topologies to Synthetic](#exporting-topologies-to-synthetic)
/// - [Retrieve distances between objects](#retrieve-distances-between-objects)
#[cfg_attr(
    feature = "hwloc-2_3_0",
    doc = "- [Comparing memory node attributes for finding where to allocate on](#comparing-memory-node-attributes-for-finding-where-to-allocate-on) (hwloc 2.3+)"
)]
#[cfg_attr(
    feature = "hwloc-2_4_0",
    doc = "- [Kinds of CPU cores](#kinds-of-cpu-cores) (hwloc 2.4+)"
)]
#[cfg_attr(
    any(doc, target_os = "linux"),
    doc = "- [Linux-specific helpers](#linux-specific-helpers)"
)]
#[cfg_attr(
    any(doc, all(target_os = "windows", feature = "hwloc-2_5_0")),
    doc = "- [Windows-specific helpers](#windows-specific-helpers) (hwloc 2.5+)"
)]
//
// --- Implementation details ---
//
// Since the Topology API is _huge_, not all of it is implemented in the
// topology module. Instead, functionality which is very strongly related to
// one other code module is implemented inside that module, leaving this module
// focused on basic lifecycle and cross-cutting issues.
//
// # Safety
//
// As a type invariant, the inner pointer is assumed to always point to a valid
// fully built, non-aliased topology.
//
// Any binding to an hwloc topology function that takes a user-provided
// &TopologyObject parameter **must** check that this object does belongs to the
// topology using the Topology::contains() method before passing it to hwloc.
#[doc(alias = "hwloc_topology")]
#[doc(alias = "hwloc_topology_t")]
pub struct Topology(NonNull<hwloc_topology>);

/// # Topology building
//
// --- Implementation details ---
//
// Upstream docs:
// - Creation: https://hwloc.readthedocs.io/en/v2.9/group__hwlocality__creation.html
// - Build queries: https://hwloc.readthedocs.io/en/v2.9/group__hwlocality__configuration.html
impl Topology {
    /// Creates a new Topology.
    ///
    /// If no customization of the build process is needed, this method is the
    /// main entry point to this crate. A topology is returned, which contains
    /// the logical representation of the physical hardware.
    ///
    /// # Examples
    ///
    /// ```
    /// # use hwlocality::Topology;
    /// let topology = Topology::new()?;
    /// # Ok::<(), eyre::Report>(())
    /// ```
    #[allow(clippy::missing_errors_doc)]
    pub fn new() -> Result<Self, RawHwlocError> {
        TopologyBuilder::new().build()
    }

    /// Test topology instance
    ///
    /// Used to avoid redundant calls to Topology::new() in unit tests and
    /// doctests that need read-only access to a topology.
    ///
    /// Configured to expose as many concepts from hwloc as possible (disallowed
    /// CPUs, I/O objects, etc).
    ///
    /// Do not use this in doctests where the fact that the topology
    /// configuration matters. And do not, under any circumstances, use this in
    /// non-test code.
    ///
    /// NOTE: In an ideal world, this would be cfg(any(test, doctest)), but that
    ///       doesn't work for doctests yet:
    ///       https://github.com/rust-lang/rust/issues/67295
    #[doc(hidden)]
    pub fn test_instance() -> &'static Self {
        static INSTANCE: OnceLock<Topology> = OnceLock::new();
        INSTANCE.get_or_init(|| {
            Self::builder()
                .with_flags(BuildFlags::INCLUDE_DISALLOWED)
                .expect("INCLUDE_DISALLOWED should always work")
                .with_common_type_filter(TypeFilter::KeepAll)
                .expect("KeepAll should be a supported common type filter")
                .build()
                .expect("Failed to initialize main test Topology")
        })
    }

    /// Like test_instance, but separate instance
    ///
    /// Used to test that operations correctly detect foreign topology objects
    #[doc(hidden)]
    pub fn foreign_instance() -> &'static Self {
        static INSTANCE: OnceLock<Topology> = OnceLock::new();
        INSTANCE.get_or_init(|| Self::test_instance().clone())
    }

    /// Prepare to create a Topology with custom configuration
    ///
    /// # Examples
    ///
    /// ```
    /// # use hwlocality::topology::{Topology, builder::BuildFlags};
    /// let flags = BuildFlags::INCLUDE_DISALLOWED;
    /// let topology = Topology::builder().with_flags(flags)?.build()?;
    /// assert_eq!(topology.build_flags(), flags);
    /// # Ok::<(), eyre::Report>(())
    /// ```
    #[doc(alias = "hwloc_topology_init")]
    pub fn builder() -> TopologyBuilder {
        TopologyBuilder::new()
    }

    /// Check that this topology is compatible with the current hwloc library
    ///
    /// This is useful when using the same topology structure (in memory) in
    /// different libraries that may use different hwloc installations (for
    /// instance if one library embeds a specific version of hwloc, while
    /// another library uses a default system-wide hwloc installation).
    ///
    /// If all libraries/programs use the same hwloc installation, this function
    /// always returns `true`.
    //
    // TODO: Propagate note about interprocess sharing from upstream docs
    //       once interprocess sharing is implemented.
    ///
    /// # Examples
    ///
    /// ```
    /// # use hwlocality::Topology;
    /// assert!(Topology::new()?.is_abi_compatible());
    /// # Ok::<(), eyre::Report>(())
    /// ```
    #[doc(alias = "hwloc_topology_abi_check")]
    pub fn is_abi_compatible(&self) -> bool {
        // SAFETY: - Topology is trusted to contain a valid ptr (type invariant)
        //         - hwloc ops are trusted not to modify *const parameters
        let result = errors::call_hwloc_int_normal("hwloc_topology_abi_check", || unsafe {
            hwlocality_sys::hwloc_topology_abi_check(self.as_ptr())
        });
        match result {
            Ok(_) => true,
            #[cfg(not(tarpaulin_include))]
            Err(RawHwlocError {
                errno: Some(Errno(EINVAL)),
                ..
            }) => false,
            #[cfg(all(windows, not(tarpaulin_include)))]
            Err(RawHwlocError { errno: None, .. }) => {
                // As explained in the RawHwlocError documentation, errno values
                // may not correctly propagate from hwloc to hwlocality on
                // Windows. Since there is only one expected errno value here,
                // we'll interpret lack of errno as EINVAL on Windows.
                false
            }
            Err(raw_err) => unreachable!("Unexpected hwloc error: {raw_err}"),
        }
    }

    /// Flags that were used to build this topology
    ///
    /// # Examples
    ///
    /// ```
    /// # use hwlocality::topology::{Topology, builder::BuildFlags};
    /// assert_eq!(Topology::new()?.build_flags(), BuildFlags::empty());
    /// # Ok::<(), eyre::Report>(())
    /// ```
    #[doc(alias = "hwloc_topology_get_flags")]
    pub fn build_flags(&self) -> BuildFlags {
        // SAFETY: - Topology is trusted to contain a valid ptr (type invariant)
        //         - hwloc ops are trusted not to modify *const parameters
        let result = BuildFlags::from_bits_truncate(unsafe {
            hwlocality_sys::hwloc_topology_get_flags(self.as_ptr())
        });
        assert!(result.is_valid(), "hwloc returned invalid flags");
        result
    }

    /// Was the topology built using the system running this program?
    ///
    /// It may not have been if, for instance, it was built using another
    /// file-system root or loaded from a synthetic or XML textual description.
    ///
    /// # Examples
    ///
    /// ```
    /// # use hwlocality::Topology;
    /// assert!(Topology::new()?.is_this_system());
    /// # Ok::<(), eyre::Report>(())
    /// ```
    #[doc(alias = "hwloc_topology_is_thissystem")]
    pub fn is_this_system(&self) -> bool {
        // SAFETY: - Topology is trusted to contain a valid ptr (type invariant)
        //         - hwloc ops are trusted not to modify *const parameters
        errors::call_hwloc_bool("hwloc_topology_is_thissystem", || unsafe {
            hwlocality_sys::hwloc_topology_is_thissystem(self.as_ptr())
        })
        .expect("Should not involve faillible syscalls")
    }

    /// Supported hwloc features with this topology on this machine
    ///
    /// This is the information that one gets via the `hwloc-info --support` CLI.
    ///
    /// The reported features are what the current topology supports on the
    /// current machine. If the topology was exported to XML from another
    /// machine and later imported here, support still describes what is
    /// supported for this imported topology after import. By default, binding
    /// will be reported as unsupported in this case (see
    /// [`BuildFlags::ASSUME_THIS_SYSTEM`]).
    ///
    #[cfg_attr(
        feature = "hwloc-2_3_0",
        doc = "On hwloc 2.3+, [`BuildFlags::IMPORT_SUPPORT`] may be used during topology building to"
    )]
    #[cfg_attr(
        feature = "hwloc-2_3_0",
        doc = "report the supported features of the original remote machine instead. If"
    )]
    #[cfg_attr(
        feature = "hwloc-2_3_0",
        doc = "it was successfully imported, [`MiscSupport::imported()`] will be set."
    )]
    ///
    /// # Examples
    ///
    /// ```
    /// # let topology = hwlocality::Topology::test_instance();
    /// println!("{:?}", topology.feature_support());
    /// # Ok::<(), eyre::Report>(())
    /// ```
    #[doc(alias = "hwloc_topology_get_support")]
    pub fn feature_support(&self) -> &FeatureSupport {
        // SAFETY: - Topology is trusted to contain a valid ptr (type invariant)
        //         - hwloc ops are trusted not to modify *const parameters
        let ptr = errors::call_hwloc_ptr("hwloc_topology_get_support", || unsafe {
            hwlocality_sys::hwloc_topology_get_support(self.as_ptr())
        })
        .expect("Unexpected hwloc error");
        // SAFETY: - If hwloc succeeded, the output is assumed to be valid and
        //           point to a valid target devoid of mutable aliases
        //         - Output reference will be bound the the lifetime of &self by
        //           the borrow checker
        unsafe { ptr.as_ref().as_newtype() }
    }

    /// Quickly check a support flag
    ///
    /// # Examples
    ///
    /// ```
    /// # use hwlocality::topology::{Topology, support::{FeatureSupport, DiscoverySupport}};
    /// let topology = Topology::new()?;
    /// assert!(topology.supports(
    ///     FeatureSupport::discovery,
    ///     DiscoverySupport::pu_count
    /// ));
    /// # Ok::<(), eyre::Report>(())
    /// ```
    pub fn supports<Group>(
        &self,
        get_group: fn(&FeatureSupport) -> Option<&Group>,
        check_feature: fn(&Group) -> bool,
    ) -> bool {
        get_group(self.feature_support()).map_or(false, check_feature)
    }

    /// Filtering that was applied for the given object type
    ///
    /// # Examples
    ///
    /// ```
    /// # use hwlocality::{
    /// #     object::types::ObjectType,
    /// #     topology::builder::TypeFilter
    /// # };
    /// # let topology = hwlocality::Topology::test_instance();
    /// #
    /// // PUs, NUMANodes and Machine are always kept
    /// let always_there = [ObjectType::PU,
    ///                     ObjectType::NUMANode,
    ///                     ObjectType::Machine];
    /// for ty in always_there {
    ///     assert_eq!(topology.type_filter(ty)?, TypeFilter::KeepAll);
    /// }
    ///
    /// // Groups are only kept if they bring extra structure
    /// assert_eq!(
    ///     topology.type_filter(ObjectType::Group)?,
    ///     TypeFilter::KeepStructure
    /// );
    /// # Ok::<(), eyre::Report>(())
    /// ```
    #[allow(clippy::missing_errors_doc)]
    #[doc(alias = "hwloc_topology_get_type_filter")]
    pub fn type_filter(&self, ty: ObjectType) -> Result<TypeFilter, RawHwlocError> {
        let mut filter = hwloc_type_filter_e::MAX;
        // SAFETY: - Topology is trusted to contain a valid ptr (type invariant)
        //         - hwloc ops are trusted not to modify *const parameters
        //         - By construction, ObjectType only exposes values that map into
        //           hwloc_obj_type_t values understood by the configured version
        //           of hwloc, and build.rs checks that the active version of
        //           hwloc is not older than that, so into() may only generate
        //           valid hwloc_obj_type_t values for current hwloc
        //         - filter is an out-parameter, initial value shouldn't matter
        errors::call_hwloc_int_normal("hwloc_topology_get_type_filter", || unsafe {
            hwlocality_sys::hwloc_topology_get_type_filter(self.as_ptr(), ty.into(), &mut filter)
        })?;
        Ok(TypeFilter::try_from(filter).expect("Unexpected type filter from hwloc"))
    }
}

/// # Distributing work items over a topology
//
// --- Implementation details ---
//
// Inspired by https://hwloc.readthedocs.io/en/v2.9/group__hwlocality__helper__distribute.html,
// but the inline header implementation had to be rewritten in Rust.
impl Topology {
    /// Distribute `num_items` work items over the topology under `roots`
    ///
    /// Given a number of work items to be processed (which can be, for example,
    /// a set of threads to be spawned), this function will assign a cpuset to
    /// each of them according to a recursive linear distribution algorithm.
    /// Such an algorithm spreads work evenly across CPUs and ensures that
    /// work-items with neighboring indices in the output array are processed by
    /// neighbouring locations in the topology, which have a high chance of
    /// sharing resources like fast CPU caches.
    ///
    /// The set of CPUs over which work items are distributed is designated by a
    /// set of root [`TopologyObject`]s with associated CPUs. The root objects
    /// must be part of this [`Topology`], or the [`ForeignRoot`] error will be
    /// returned. Their CPU sets must also be disjoint, or the
    /// [`OverlappingRoots`] error will be returned. You can distribute items
    /// across all CPUs in the topology by setting `roots` to
    /// `&[topology.root_object()]`.
    ///
    /// Since the purpose of `roots` is to designate which CPUs items should be
    /// allocated to, root objects should normally have a CPU set. If that is
    /// not the case (e.g. if some roots designate NUMA nodes or I/O objects
    /// like storage or GPUs), the algorithm will walk up affected roots'
    /// ancestor chains to locate the first ancestor with CPUs in the topology,
    /// which represents the CPUs closest to the object of interest. If none of
    /// the CPUs of that ancestor is available for binding, that root will be
    /// ignored, unless this is true of all roots in which case the
    /// [`EmptyRoots`] error is returned.
    ///
    /// If there is no depth limit, which can be achieved by setting `max_depth`
    /// to [`NormalDepth::MAX`], the distribution will be done down to the
    /// granularity of individual CPUs, i.e. if there are more work items that
    /// CPUs, each work item will be assigned one CPU. By setting the
    /// `max_depth` parameter to a lower limit, you can distribute work at a
    /// coarser granularity, e.g. across L3 caches, giving the OS some leeway to
    /// move tasks across CPUs sharing that cache.
    ///
    /// By default, output cpusets follow the logical topology children order.
    /// By setting `flags` to [`DistributeFlags::REVERSE`], you can ask for them
    /// to be provided in reverse order instead (from last child to first child).
    ///
    /// # Errors
    ///
    /// - [`EmptyRoots`] if there are no CPUs to distribute work to (the
    ///   union of all root cpusets is empty).
    /// - [`ForeignRoot`] if some of the specified roots do not belong to this
    ///   topology.
    /// - [`OverlappingRoots`] if some of the roots have overlapping CPU sets.
    ///
    /// [`EmptyRoots`]: DistributeError::EmptyRoots
    /// [`ForeignRoot`]: DistributeError::ForeignRoot
    /// [`OverlappingRoots`]: DistributeError::OverlappingRoots
    #[allow(clippy::missing_docs_in_private_items)]
    #[doc(alias = "hwloc_distrib")]
    pub fn distribute_items(
        &self,
        roots: &[&TopologyObject],
        num_items: usize,
        max_depth: NormalDepth,
        flags: DistributeFlags,
    ) -> Result<Vec<CpuSet>, DistributeError> {
        // Make sure all roots belong to this topology
        for root in roots.iter().copied() {
            if !self.contains(root) {
                return Err(DistributeError::ForeignRoot(root.into()));
            }
        }

        // Handle the trivial case where 0 items are distributed
        if num_items == 0 {
            return Ok(Vec::new());
        }

        /// Inner recursive distribution algorithm
        fn recurse<'a>(
            roots_and_cpusets: impl DoubleEndedIterator<Item = ObjSetWeightDepth<'a>> + Clone,
            num_items: usize,
            max_depth: NormalDepth,
            flags: DistributeFlags,
            result: &mut Vec<CpuSet>,
        ) {
            // Debug mode checks
            debug_assert_ne!(
                roots_and_cpusets.clone().count(),
                0,
                "Can't distribute to 0 roots"
            );
            debug_assert_ne!(
                num_items, 0,
                "Shouldn't try to distribute 0 items (just don't call this function)"
            );
            let initial_len = result.len();

            // Total number of cpus covered by the active roots
            let total_weight: usize = roots_and_cpusets
                .clone()
                .map(|(_, _, weight, _)| weight)
                .sum();

            // Subset of CPUs and items covered so far
            let mut given_weight = 0;
            let mut given_items = 0;

            // What to do with each root
            let process_root = |(root, cpuset, weight, depth): ObjSetWeightDepth<'_>| {
                // Give this root a chunk of the work-items proportional to its
                // weight, with a bias towards giving more CPUs to first roots
                let next_given_weight = given_weight + weight;
                let next_given_items = weight_to_items(next_given_weight, total_weight, num_items);
                let my_items = next_given_items - given_items;

                // Keep recursing until we reach the bottom of the topology,
                // run out of items to distribute, or hit the depth limit
                if root.normal_arity() > 0 && my_items > 1 && depth < max_depth {
                    recurse(
                        root.normal_children().filter_map(decode_normal_obj),
                        my_items,
                        max_depth,
                        flags,
                        result,
                    );
                } else if my_items > 0 {
                    // All items attributed to this root get this root's cpuset
                    for _ in 0..my_items {
                        result.push(cpuset.clone_target());
                    }
                } else {
                    // No item attributed to this root, merge cpuset with
                    // the previous root.
                    let mut iter = result.iter_mut().rev();
                    let last = iter.next().expect("First chunk cannot be empty");
                    for other in iter {
                        if other == last {
                            *other |= cpuset;
                        }
                    }
                    *last |= cpuset;
                }

                // Prepare to process the next root
                given_weight = next_given_weight;
                given_items = next_given_items;
            };

            // Process roots in the order dictated by flags
            if flags.contains(DistributeFlags::REVERSE) {
                roots_and_cpusets.rev().for_each(process_root);
            } else {
                roots_and_cpusets.for_each(process_root);
            }

            // Debug mode checks
            debug_assert_eq!(
                result.len() - initial_len,
                num_items,
                "This function should distribute the requested number of items"
            );
        }

        // Check roots, walk up to their first normal ancestor as necessary
        let decoded_roots = roots.iter().copied().filter_map(|root| {
            let mut root_then_ancestors = std::iter::once(root)
                .chain(root.ancestors())
                .skip_while(|candidate| !candidate.object_type().is_normal());
            root_then_ancestors.find_map(decode_normal_obj)
        });
        if decoded_roots.clone().count() == 0 {
            return Err(DistributeError::EmptyRoots);
        }
        if sets_overlap(decoded_roots.clone().map(|(_, root_set, _, _)| root_set)) {
            return Err(DistributeError::OverlappingRoots);
        }

        // Run the recursion, collect results
        let mut result = Vec::with_capacity(num_items);
        recurse(decoded_roots, num_items, max_depth, flags, &mut result);
        debug_assert_eq!(
            result.len(),
            num_items,
            "This function should produce one result per input item"
        );
        Ok(result)
    }
}
//
#[cfg(not(tarpaulin_include))]
bitflags! {
    /// Flags to be given to [`Topology::distribute_items()`]
    #[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
    #[doc(alias = "hwloc_distrib_flags_e")]
    pub struct DistributeFlags: hwloc_distrib_flags_e {
        /// Distribute in reverse order, starting from the last objects
        const REVERSE = HWLOC_DISTRIB_FLAG_REVERSE;
    }
}
//
crate::impl_arbitrary_for_bitflags!(DistributeFlags, hwloc_distrib_flags_e);
//
impl Default for DistributeFlags {
    fn default() -> Self {
        Self::empty()
    }
}
//
/// Error returned by [`Topology::distribute_items()`]
#[derive(Clone, Debug, Eq, Error, Hash, PartialEq)]
pub enum DistributeError {
    /// Error returned when the specified roots contain no accessible CPUs
    ///
    /// This can happen if either an empty roots list is specified, or if the
    /// topology was built with [`BuildFlags::INCLUDE_DISALLOWED`] and the
    /// specified roots only contain disallowed CPUs.
    #[error("no CPU is accessible from the distribution roots")]
    EmptyRoots,

    /// Some of the specified roots do not belong to this topology
    #[error("distribution root {0}")]
    ForeignRoot(#[from] ForeignObjectError),

    /// Specified roots overlap ith each other
    #[error("distribution roots overlap with each other")]
    OverlappingRoots,
}

/// Part of the implementation of [`Topology::distribute_items()`] that tells,
/// given a number of items to distribute, a cpuset weight, and the sum of all
/// cpuset weights, how many items should be distributed, rounding up.
fn weight_to_items(given_weight: usize, total_weight: usize, num_items: usize) -> usize {
    // Weights represent CPU counts and num_items represents
    // something that users reasonably want to count. Neither
    // should overflow u128 even on highly speculative future
    // hardware where usize would be larger than 128 bits
    #[allow(clippy::missing_docs_in_private_items)]
    const TOO_LARGE: &str = "Such large inputs are not supported yet";
    let cast = |x: usize| u128::try_from(x).expect(TOO_LARGE);

    // Numerator product can only fail if both given_weight and
    // num_items are >=2^64, which is equally improbable.
    let numerator = cast(given_weight)
        .checked_mul(cast(num_items))
        .expect(TOO_LARGE);
    let denominator = cast(total_weight);
    let my_items = numerator / denominator + u128::from(numerator % denominator != 0);

    // Cast can only fail if given_weight > tot_weight,
    // otherwise we expect my_items <= num_items
    debug_assert!(
        given_weight <= total_weight,
        "Cannot distribute more weight than the active root's total weight"
    );
    my_items
        .try_into()
        .expect("Cannot happen if computation is correct")
}

/// Part of the implementation of [`Topology::distribute_items()`] that extracts
/// information from a [`TopologyObject`] that is known to be normal, and
/// returns this information if the object has a non-empty cpuset
fn decode_normal_obj(obj: &TopologyObject) -> Option<ObjSetWeightDepth<'_>> {
    debug_assert!(
        obj.object_type().is_normal(),
        "This function only works on normal objects"
    );
    let cpuset = obj.cpuset().expect("Normal objects should have cpusets");
    let weight = cpuset
        .weight()
        .expect("Topology objects should not have infinite cpusets");
    let depth = obj.depth().expect_normal();
    (weight > 0).then_some((obj, cpuset, weight, depth))
}

/// Information that is extracted by [`decode_normal_obj()`]
type ObjSetWeightDepth<'a> = (
    &'a TopologyObject,
    BitmapRef<'a, CpuSet>,
    usize,
    NormalDepth,
);

/// Truth that an iterator of cpusets contains overlapping sets
///
/// `sets` can yield `&'_ CpuSet` or `BitmapRef<'_, CpuSet>`.
fn sets_overlap(mut sets: impl Iterator<Item = impl Deref<Target = CpuSet>>) -> bool {
    sets.try_fold(CpuSet::new(), |mut acc, set| {
        let set: &CpuSet = &set;
        if acc.intersects(set) {
            None
        } else {
            acc |= set;
            Some(acc)
        }
    })
    .is_none()
}

/// # CPU and node sets of entire topologies
//
// --- Implementation details ---
//
// Upstream docs: https://hwloc.readthedocs.io/en/v2.9/group__hwlocality__helper__topology__sets.html
impl Topology {
    /// Topology CPU set
    ///
    /// This is equivalent to calling [`TopologyObject::cpuset()`] on
    /// the topology's root object.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use hwlocality::Topology;
    /// # let topology = Topology::test_instance();
    /// println!("Visible CPUs in this topology: {}", topology.cpuset());
    /// # Ok::<_, eyre::Report>(())
    /// ```
    #[doc(alias = "hwloc_topology_get_topology_cpuset")]
    pub fn cpuset(&self) -> BitmapRef<'_, CpuSet> {
        // SAFETY: This is one of the hwloc functions allowed here
        unsafe {
            self.topology_set(
                "hwloc_topology_get_topology_cpuset",
                hwlocality_sys::hwloc_topology_get_topology_cpuset,
            )
        }
    }

    /// Complete CPU set
    ///
    /// This is equivalent to calling [`TopologyObject::complete_cpuset()`] on
    /// the topology's root object.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use hwlocality::Topology;
    /// # let topology = Topology::test_instance();
    /// println!(
    ///     "Overall CPUs in this topology: {}",
    ///     topology.complete_cpuset()
    /// );
    /// # Ok::<_, eyre::Report>(())
    /// ```
    #[doc(alias = "hwloc_topology_get_complete_cpuset")]
    pub fn complete_cpuset(&self) -> BitmapRef<'_, CpuSet> {
        // SAFETY: This is one of the hwloc functions allowed here
        unsafe {
            self.topology_set(
                "hwloc_topology_get_complete_cpuset",
                hwlocality_sys::hwloc_topology_get_complete_cpuset,
            )
        }
    }

    /// Allowed CPU set
    ///
    /// If [`BuildFlags::INCLUDE_DISALLOWED`] was not set, this is identical to
    /// [`Topology::cpuset()`]: all visible PUs are allowed.
    ///
    /// Otherwise, you can check whether a particular cpuset contains allowed
    /// PUs by calling `cpuset.intersects(topology.allowed_cpuset())`, and if so
    /// you can get the set of allowed PUs with
    /// `cpuset & topology.allowed_cpuset()`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use hwlocality::Topology;
    /// # let topology = Topology::test_instance();
    /// println!(
    ///     "Allowed CPUs in this topology: {}",
    ///     topology.allowed_cpuset()
    /// );
    /// # Ok::<_, eyre::Report>(())
    /// ```
    #[doc(alias = "hwloc_topology_get_allowed_cpuset")]
    pub fn allowed_cpuset(&self) -> BitmapRef<'_, CpuSet> {
        // SAFETY: This is one of the hwloc functions allowed here
        unsafe {
            self.topology_set(
                "hwloc_topology_get_allowed_cpuset",
                hwlocality_sys::hwloc_topology_get_allowed_cpuset,
            )
        }
    }

    /// Topology node set
    ///
    /// This is equivalent to calling [`TopologyObject::nodeset()`] on
    /// the topology's root object.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use hwlocality::Topology;
    /// # let topology = Topology::test_instance();
    /// println!("Visible NUMA nodes in this topology: {}", topology.nodeset());
    /// # Ok::<_, eyre::Report>(())
    /// ```
    #[doc(alias = "hwloc_topology_get_topology_nodeset")]
    pub fn nodeset(&self) -> BitmapRef<'_, NodeSet> {
        // SAFETY: This is one of the hwloc functions allowed here
        unsafe {
            self.topology_set(
                "hwloc_topology_get_topology_nodeset",
                hwlocality_sys::hwloc_topology_get_topology_nodeset,
            )
        }
    }

    /// Complete node set
    ///
    /// This is equivalent to calling [`TopologyObject::complete_nodeset()`] on
    /// the topology's root object.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use hwlocality::Topology;
    /// # let topology = Topology::test_instance();
    /// println!(
    ///     "Overall NUMA nodes in this topology: {}",
    ///     topology.complete_nodeset()
    /// );
    /// # Ok::<_, eyre::Report>(())
    /// ```
    #[doc(alias = "hwloc_topology_get_complete_nodeset")]
    pub fn complete_nodeset(&self) -> BitmapRef<'_, NodeSet> {
        // SAFETY: This is one of the hwloc functions allowed here
        unsafe {
            self.topology_set(
                "hwloc_topology_get_complete_nodeset",
                hwlocality_sys::hwloc_topology_get_complete_nodeset,
            )
        }
    }

    /// Allowed node set
    ///
    /// If [`BuildFlags::INCLUDE_DISALLOWED`] was not set, this is identical to
    /// [`Topology::nodeset()`]: all visible NUMA nodes are allowed.
    ///
    /// Otherwise, you can check whether a particular nodeset contains allowed
    /// NUMA nodes by calling `nodeset.intersects(topology.allowed_nodeset())`,
    /// and if so you can get the set of allowed NUMA nodes with
    /// `nodeset & topology.allowed_nodeset()`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use hwlocality::Topology;
    /// # let topology = Topology::test_instance();
    /// println!(
    ///     "Allowed NUMA nodes in this topology: {}",
    ///     topology.allowed_nodeset()
    /// );
    /// # Ok::<_, eyre::Report>(())
    /// ```
    #[doc(alias = "hwloc_topology_get_allowed_nodeset")]
    pub fn allowed_nodeset(&self) -> BitmapRef<'_, NodeSet> {
        // SAFETY: This is one of the hwloc functions allowed here
        unsafe {
            self.topology_set(
                "hwloc_topology_get_allowed_nodeset",
                hwlocality_sys::hwloc_topology_get_allowed_nodeset,
            )
        }
    }

    /// Query a topology-wide `CpuSet` or `NodeSet`
    ///
    /// # Safety
    ///
    /// `getter` must be one of the functions described in the ["CPU and node
    /// sets of entire
    /// topologies"](https://hwloc.readthedocs.io/en/v2.9/group__hwlocality__helper__topology__sets.html)
    /// section of the hwloc documentation, which means in particular that it...
    ///
    /// - Cannot return NULL
    /// - Must return a pointer attached to the topology
    unsafe fn topology_set<'topology, Set: SpecializedBitmap>(
        &'topology self,
        getter_name: &'static str,
        getter: unsafe extern "C" fn(*const hwloc_topology) -> *const hwloc_bitmap_s,
    ) -> BitmapRef<'topology, Set> {
        // SAFETY: - Topology is trusted to contain a valid ptr (type invariant)
        //         - hwloc ops are trusted not to modify *const parameters
        //         - If this operation is successful, it should return a valid
        //           bitmap pointer
        //         - Output bitmap is indeed bound to the topology's lifetime
        let bitmap_ref = unsafe {
            let bitmap_ptr = errors::call_hwloc_ptr(getter_name, || getter(self.as_ptr()))
                .expect("According to their docs, these functions cannot return NULL");
            Bitmap::borrow_from_nonnull::<'topology>(bitmap_ptr)
        };
        bitmap_ref.cast()
    }
}

// # General-purpose internal utilities
impl Topology {
    /// Contained hwloc topology pointer (for interaction with hwloc)
    pub(crate) fn as_ptr(&self) -> *const hwloc_topology {
        self.0.as_ptr()
    }

    /// Contained mutable hwloc topology pointer (for interaction with hwloc)
    ///
    /// Be warned that as a result of hwloc employing lazy caching techniques,
    /// almost every interaction that requires `*mut hwloc_topology` is unsafe
    /// unless followed by `hwloc_topology_refresh()`. This subtlety is handled
    /// by the [`Topology::edit()`] mechanism.
    pub(crate) fn as_mut_ptr(&mut self) -> *mut hwloc_topology {
        self.0.as_ptr()
    }

    /// Check if a [`TopologyObject`] is part of this topology
    ///
    /// This check is a safety precondition to any hwloc topology method
    /// binding that takes user-originated `&TopologyObject`s as a parameter
    /// and passes it to an hwloc topology method.
    ///
    /// While this is not expected to happen often and will in fact often
    /// require the user to jump through some serious hoops like creating
    /// another `static` or `thread_local` topology, unfortunately there is
    /// always a way to do it, and safe Rust code must remain safe even in the
    /// craziest of edge cases...
    pub(crate) fn contains(&self, object: &TopologyObject) -> bool {
        let expected_root = self.root_object();
        let actual_root = std::iter::once(object)
            .chain(object.ancestors())
            .last()
            .expect("By definition, this iterator always has >= 1 element");
        std::ptr::eq(expected_root, actual_root)
    }
}

impl Clone for Topology {
    #[doc(alias = "hwloc_topology_dup")]
    fn clone(&self) -> Self {
        let mut clone = ptr::null_mut();
        // SAFETY: - Topology is trusted to contain a valid ptr (type invariant)
        //         - hwloc ops are trusted not to modify *const parameters
        //         - clone is an out-parameter, it can have any initial value
        errors::call_hwloc_int_normal("hwloc_topology_dup", || unsafe {
            hwlocality_sys::hwloc_topology_dup(&mut clone, self.as_ptr())
        })
        .expect("Duplicating a topology should not fail");

        Self(NonNull::new(clone).expect("Got null pointer from hwloc_topology_dup"))
    }
}

impl Debug for Topology {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut debug = f.debug_struct("Topology");

        // Topology building properties
        debug
            .field("is_abi_compatible", &self.is_abi_compatible())
            .field("build_flags", &self.build_flags())
            .field("is_this_system", &self.is_this_system())
            .field("feature_support", self.feature_support());
        let type_filters = enum_iterator::all::<ObjectType>()
            .map(|ty| {
                (
                    format!("{ty}"),
                    self.type_filter(ty)
                        .expect("should always succeed when called with a valid type"),
                )
            })
            .collect::<BTreeMap<_, _>>();
        debug.field("type_filter", &type_filters);

        // TopologyObject hierarchy
        let objects_per_depth = NormalDepth::iter_range(NormalDepth::MIN, self.depth())
            .map(Depth::from)
            .chain(Depth::VIRTUAL_DEPTHS.iter().copied())
            .filter_map(|depth| {
                let objs = self.objects_at_depth(depth).collect::<Vec<_>>();
                (!objs.is_empty()).then_some((format!("{depth}"), objs))
            })
            .collect::<Vec<_>>();
        debug
            // Contains the info from most other topology queries
            .field("objects_per_depth", &objects_per_depth)
            .field("memory_parents_depth", &self.memory_parents_depth());

        // CPU and node sets of the entire topology
        debug
            .field("cpuset", &self.cpuset())
            .field("complete_cpuset", &self.complete_cpuset())
            .field("allowed_cpuset", &self.allowed_cpuset())
            .field("nodeset", &self.nodeset())
            .field("complete_nodeset", &self.complete_nodeset())
            .field("allowed_nodeset", &self.allowed_nodeset());

        // Object distances
        debug.field("distances", &self.distances(None));

        // Memory attributes
        #[cfg(feature = "hwloc-2_3_0")]
        {
            debug.field("memory_attributes", &self.dump_memory_attributes());
        }

        // Kinds of CPU cores
        #[cfg(feature = "hwloc-2_4_0")]
        {
            let cpu_kinds = self.cpu_kinds().into_iter().flatten().collect::<Vec<_>>();
            debug.field("cpu_kinds", &cpu_kinds);
        }
        debug.finish()
    }
}

impl Drop for Topology {
    #[doc(alias = "hwloc_topology_destroy")]
    fn drop(&mut self) {
        // SAFETY: - Topology is trusted to contain a valid ptr (type invariant)
        //         - Topology will not be usable again after Drop
        unsafe { hwlocality_sys::hwloc_topology_destroy(self.as_mut_ptr()) }
    }
}

impl PartialEq for Topology {
    fn eq(&self, other: &Self) -> bool {
        /// Check equality of basic topology properties
        macro_rules! check_all_equal {
            ($($property:ident),*) => {
                if (
                    $(
                        Topology::$property(self) != Topology::$property(other)
                    )||*
                ) {
                    return false;
                }
            };
        }
        check_all_equal!(
            is_abi_compatible,
            build_flags,
            is_this_system,
            feature_support,
            memory_parents_depth,
            cpuset,
            complete_cpuset,
            allowed_cpuset,
            nodeset,
            complete_nodeset,
            allowed_nodeset
        );

        /// Retrieve all type filters to check they are the same
        fn type_filters(
            topology: &Topology,
        ) -> impl Iterator<Item = Result<TypeFilter, RawHwlocError>> + '_ {
            enum_iterator::all::<ObjectType>().map(|ty| topology.type_filter(ty))
        }
        if !type_filters(self).eq(type_filters(other)) {
            return false;
        }

        // Check that the object hierarchy is the same
        if !self.has_same_object_hierarchy(other) {
            return false;
        }

        // Check that object distances are the same
        let same_distances = match (self.distances(None), other.distances(None)) {
            (Ok(distances1), Ok(distances2)) => {
                distances1.len() == distances2.len()
                    && distances1
                        .into_iter()
                        .zip(&distances2)
                        .all(|(d1, d2)| d1.eq_modulo_topology(d2))
            }
            (Err(e1), Err(e2)) => e1 == e2,
            (Ok(_), Err(_)) | (Err(_), Ok(_)) => false,
        };
        if !same_distances {
            return false;
        }

        // Check that memory attributes are the same
        #[cfg(feature = "hwloc-2_3_0")]
        {
            if !self
                .dump_memory_attributes()
                .eq_modulo_topology(&other.dump_memory_attributes())
            {
                return false;
            }
        }

        // Check that CPU core kinds are the same
        #[cfg(feature = "hwloc-2_4_0")]
        {
            let same_kinds = match (self.cpu_kinds(), other.cpu_kinds()) {
                (Ok(kinds1), Ok(kinds2)) => kinds1.eq(kinds2),
                (Err(e1), Err(e2)) => e1 == e2,
                (Ok(_), Err(_)) | (Err(_), Ok(_)) => false,
            };
            if !same_kinds {
                return false;
            }
        }
        true
    }
}

impl Pointer for Topology {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <NonNull<hwloc_topology> as Pointer>::fmt(&self.0, f)
    }
}

// SAFETY: No shared mutability
unsafe impl Send for Topology {}

// SAFETY: No shared mutability
unsafe impl Sync for Topology {}

#[allow(clippy::too_many_lines)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ffi::PositiveInt, topology::builder::tests::DataSource};
    use bitflags::Flags;
    use proptest::prelude::*;
    #[allow(unused)]
    use similar_asserts::assert_eq;
    use static_assertions::{assert_impl_all, assert_not_impl_any};
    use std::{
        collections::BTreeSet,
        error::Error,
        fmt::{Binary, Display, LowerExp, LowerHex, Octal, UpperExp, UpperHex},
        hash::Hash,
        io::{self, Read},
        num::NonZeroU8,
        ops::{
            BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not, Sub, SubAssign,
        },
        panic::UnwindSafe,
    };

    // Check that public types in this module keep implementing all expected
    // traits, in the interest of detecting future semver-breaking changes
    assert_impl_all!(DistributeFlags:
        Binary, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign,
        Copy, Debug, Default, Extend<DistributeFlags>, Flags,
        FromIterator<DistributeFlags>, Hash, IntoIterator<Item=DistributeFlags>,
        LowerHex, Not, Octal, Sized, Sub, SubAssign, Sync, UpperHex, Unpin,
        UnwindSafe
    );
    assert_not_impl_any!(DistributeFlags:
        Display, Drop, PartialOrd, Pointer, LowerExp, Read, UpperExp,
        fmt::Write, io::Write
    );
    assert_impl_all!(Topology:
        Clone, Debug, Drop, PartialEq, Pointer, Sized, Sync, Unpin, UnwindSafe
    );
    assert_not_impl_any!(Topology:
        Binary, Copy, Default, Deref, Display, IntoIterator, LowerExp, LowerHex,
        Octal, Read, UpperExp, UpperHex, fmt::Write, io::Write
    );
    assert_impl_all!(DistributeError:
        Clone, Error, Hash, Sized, Sync, Unpin, UnwindSafe
    );
    assert_not_impl_any!(DistributeError:
        Binary, Copy, Default, Deref, Drop, IntoIterator,
        LowerExp, LowerHex, Octal, PartialOrd, Pointer, Read,
        UpperExp, UpperHex, fmt::Write, io::Write
    );

    #[test]
    fn default_distribute_flags() {
        assert_eq!(DistributeFlags::default(), DistributeFlags::empty());
    }

    #[test]
    fn clone() -> Result<(), TestCaseError> {
        let topology = Topology::test_instance();
        let clone = topology.clone();
        assert_ne!(format!("{:p}", *topology), format!("{clone:p}"));
        builder::tests::check_topology(
            topology,
            DataSource::ThisSystem,
            topology.build_flags(),
            |ty| Ok(topology.type_filter(ty).unwrap()),
        )?;
        assert!(!topology.contains(clone.root_object()));
        assert_eq!(topology, &clone);
        Ok(())
    }

    #[allow(clippy::print_stdout, clippy::use_debug)]
    #[test]
    fn debug_and_self_eq() {
        let topology = Topology::test_instance();
        assert_eq!(topology, topology);
        println!("{topology:#?}");
    }

    /// Bias the `max_depth` input to `distribute_items` tests so that
    /// interesting depth values below the maximum possible depth are sampled
    /// often enough
    fn max_depth() -> impl Strategy<Value = NormalDepth> {
        prop_oneof![
            4 => (0..usize::from(Topology::test_instance().depth()))
                    .prop_map(|us| NormalDepth::try_from(us).unwrap()),
            1 => any::<NormalDepth>()
        ]
    }

    proptest! {
        // Check that absence of roots to distribute too is reported correctly
        #[test]
        fn distribute_nowhere(num_items: NonZeroU8, max_depth in max_depth(), flags: DistributeFlags) {
            let num_items = usize::from(num_items.get());
            prop_assert_eq!(
                Topology::test_instance().distribute_items(&[], num_items, max_depth, flags),
                Err(DistributeError::EmptyRoots)
            );
        }
    }

    /// Generate valid (disjoint) roots for [`Topology::distribute_items()`],
    /// taken from [`Topology::test_instance()`]
    fn disjoint_roots() -> impl Strategy<Value = Vec<&'static TopologyObject>> {
        /// Number of PUs below a normal object
        fn normal_weight(obj: &TopologyObject) -> usize {
            obj.cpuset()
                .expect("normal object should have a cpuset")
                .weight()
                .expect("normal object should have a finite cpuset")
        }

        /// Pick N objects below a certain root
        ///
        /// `root` must be a normal object that has at least `num_objects` PUs
        /// underneath it.
        ///
        /// Objects will be ordered by logical index, the client is responsible
        /// for shuffling them at the very end.
        fn pick_disjoint_objects(
            root: &'static TopologyObject,
            num_objects: usize,
        ) -> impl Strategy<Value = Vec<&'static TopologyObject>> {
            // Validate the request
            assert!(
                root.object_type().is_normal(),
                "root object should be normal"
            );
            assert!(num_objects <= normal_weight(root));

            // Honor the request
            match num_objects {
                // Picking no objects is trivial
                0 => Just(Vec::new()).boxed(),

                // Picking a single object is easy too, just pick any object in
                // the subtree below this root
                1 => {
                    let topology = Topology::test_instance();
                    let subtree_objects = topology
                        .normal_objects()
                        .filter(|obj| obj.is_in_subtree(root) || ptr::eq(*obj, root))
                        .collect::<Vec<_>>();
                    prop::sample::select(subtree_objects)
                        .prop_map(|obj| vec![obj])
                        .boxed()
                }

                // If we need to pick 2 or more objects, we must distribute them
                // across the root's children
                _ => {
                    // Each child can at most yield as many objects as there are
                    // PUs underneath it, and we must not yield more PUs than
                    // this. This is achieved through the dirty trick of first
                    // duplicating each child once per underlying PU...
                    let degenerate_children = root
                        .normal_children()
                        .flat_map(|child| std::iter::repeat(child).take(normal_weight(child)))
                        .collect::<Vec<_>>();
                    debug_assert_eq!(degenerate_children.len(), normal_weight(root));

                    // Then picking a subsequence of that duplicated sequence...
                    prop::sample::subsequence(degenerate_children, num_objects)
                        .prop_flat_map(|selected_degenerate| {
                            // Then deduplicating again to find out how many
                            // objects were allocated to each child.
                            let mut count_per_child =
                                Vec::<(&'static TopologyObject, usize)>::new();
                            'children: for child in selected_degenerate {
                                if let Some((last_child, last_count)) = count_per_child.last_mut() {
                                    if ptr::eq(child, *last_child) {
                                        *last_count += 1;
                                        continue 'children;
                                    }
                                }
                                count_per_child.push((child, 1));
                            }

                            // Once we have the deduplicated per-child work
                            // allocation, we do the recursive request, which
                            // will yield a vector of results for each child...
                            let nested_objs = count_per_child
                                .into_iter()
                                .map(|(child, count)| pick_disjoint_objects(child, count))
                                .collect::<Vec<_>>();

                            // ...and we flatten that into a single vector of
                            // merged results
                            nested_objs.prop_map(|nested_objs| {
                                nested_objs.into_iter().flatten().collect::<Vec<_>>()
                            })
                        })
                        .boxed()
                }
            }
        }

        // Finally, we use the above logic to generate any valid number of roots
        let root = Topology::test_instance().root_object();
        (1..=normal_weight(root))
            .prop_flat_map(|num_objects| pick_disjoint_objects(root, num_objects))
            .prop_shuffle()
    }

    proptest! {
        // Check that requests to distribute 0 items are handled correctly
        #[test]
        fn distribute_nothing(
            disjoint_roots in disjoint_roots(),
            max_depth in max_depth(),
            flags: DistributeFlags,
        ) {
            prop_assert_eq!(
                Topology::test_instance().distribute_items(&disjoint_roots, 0, max_depth, flags),
                Ok(Vec::new())
            );
        }
    }

    /// Provide a pessimistic set of leaves amongst which
    /// [`Topology::distribute_items()`] could distribute work, given roots and
    /// a max depth and a number of items to be distributed
    ///
    /// The actual set can be more restrained in the presence of heterogeneous
    /// root objects.
    fn find_possible_leaves<'output>(
        roots: &[&'output TopologyObject],
        max_depth: NormalDepth,
    ) -> Vec<&'output TopologyObject> {
        let mut input = roots.to_vec();
        let mut output = Vec::new();
        for _ in PositiveInt::iter_range(PositiveInt::MIN, max_depth) {
            let mut new_leaves = false;
            for obj in input.drain(..) {
                if obj.normal_arity() > 0 && obj.depth().expect_normal() < max_depth {
                    output.extend(obj.normal_children());
                    new_leaves = true;
                } else {
                    output.push(obj);
                }
            }
            std::mem::swap(&mut input, &mut output);
            if !new_leaves {
                break;
            }
        }
        input
    }

    proptest! {
        // Check that distribute_items works well given correct inputs
        #[test]
        fn distribute_correct(
            disjoint_roots in disjoint_roots(),
            max_depth in max_depth(),
            num_items: NonZeroU8,
            flags: DistributeFlags,
        ) {
            // Set up test state
            let topology = Topology::test_instance();
            let num_items = usize::from(num_items.get());

            // Check results on normal roots
            #[allow(clippy::cast_precision_loss)]
            {
                // Run the computation
                let item_sets = topology
                    .distribute_items(&disjoint_roots, num_items, max_depth, flags)
                    .unwrap();
                prop_assert_eq!(item_sets.len(), num_items);

                // Predict among which leaves of the object tree work items could
                // potentially have been distributed
                let possible_leaf_objects = find_possible_leaves(&disjoint_roots, max_depth);
                let mut possible_leaf_sets = possible_leaf_objects
                    .iter()
                    .map(|obj| obj.cpuset().unwrap().clone_target())
                    .collect::<BTreeSet<CpuSet>>();

                // Count how many work items were distributed to each leaf
                let mut items_per_set = Vec::<(CpuSet, usize)>::new();
                for item_set in item_sets {
                    match items_per_set.last_mut() {
                        Some((last_set, count)) if last_set == &item_set => *count += 1,
                        _ => items_per_set.push((item_set, 1)),
                    }
                }

                // Make sure work items were indeed only distributed to these leaves
                for item_set in items_per_set.iter().map(|(set, _count)| set) {
                    // ...but bear in mind that the algorithm may merge adjacent
                    // leaf cpusets if there are fewer items than leaf cpusets.
                    if !possible_leaf_sets.contains(item_set) {
                        // Compute the expected effect of merging
                        let subsets = possible_leaf_objects
                            .iter()
                            .map(|obj| obj.cpuset().unwrap())
                            .skip_while(|&subset| !item_set.includes(subset))
                            .take_while(|&subset| item_set.includes(subset))
                            .map(|subset| subset.clone_target())
                            .collect::<Vec<_>>();
                        let merged_set = subsets.iter().fold(CpuSet::new(), |mut acc, subset| {
                            acc |= subset;
                            acc
                        });

                        // Check if this indeed what happened
                        prop_assert_eq!(
                            item_set,
                            &merged_set,
                            "Distributed set {} is not part of expected leaf sets {:?}",
                            item_set,
                            possible_leaf_sets
                        );
                        prop_assert_eq!(
                            items_per_set
                                .iter()
                                .filter(|(item_set, _count)| item_set.intersects(&merged_set))
                                .count(),
                            1,
                            "Merging should only occurs when 1 item is shared by N leaves"
                        );

                        // Take the merging of leaves into account
                        for subset in subsets {
                            prop_assert!(possible_leaf_sets.remove(&subset));
                        }
                        prop_assert!(possible_leaf_sets.insert(merged_set));
                    }
                }

                // Make sure the leaves that were actually used are disjoint (i.e.
                // no distributing to a parent AND its children)
                prop_assert!(!sets_overlap(items_per_set.iter().map(|(set, _count)| set)));

                // Check that the distribution is as close to ideal as possible,
                // given that work items cannot be split in halves
                let total_weight = items_per_set
                    .iter()
                    .map(|(leaf_set, _count)| leaf_set.weight().unwrap())
                    .sum::<usize>();
                for (leaf_set, items_per_leaf) in &items_per_set {
                    let cpu_share = leaf_set.weight().unwrap() as f64 / total_weight as f64;
                    let ideal_share = num_items as f64 * cpu_share;
                    prop_assert!((*items_per_leaf as f64 - ideal_share).abs() <= 1.0);
                }

                // Check that the distribution is biased towards earlier or later
                // leaves depending on distribution flags, due to roots being
                // enumerated in reverse order
                let (first_set, items_per_leaf) = items_per_set.first().unwrap();
                let first_set_intersects = |root: Option<&TopologyObject>| {
                    Ok(prop_assert!(first_set.intersects(root.unwrap().cpuset().unwrap())))
                };
                if flags.contains(DistributeFlags::REVERSE) {
                    first_set_intersects(disjoint_roots.last().copied())?;
                } else {
                    first_set_intersects(disjoint_roots.first().copied())?;
                }
                let cpu_share = first_set.weight().unwrap() as f64 / total_weight as f64;
                let ideal_share = num_items as f64 * cpu_share;
                let bias = *items_per_leaf as f64 - ideal_share;
                prop_assert!(
                    bias >= 0.0,
                    "Earlier roots should get favored for item allocation"
                );
            }
        }
    }

    /// To the random input provided by `disjoint_roots`, add an extra root that
    /// overlaps with the existing ones
    fn overlapping_roots() -> impl Strategy<Value = Vec<&'static TopologyObject>> {
        disjoint_roots().prop_flat_map(|disjoint_roots| {
            // Randomly pick a CPU in the set covered by the disjoint roots, and
            // find the smallest hwloc object (normally a PU) that covers it.
            // This will be used to check distribute_items's handling of
            // duplicate roots.
            let topology = Topology::test_instance();
            let full_set = disjoint_roots.iter().fold(CpuSet::new(), |mut acc, root| {
                acc |= root.cpuset().unwrap();
                acc
            });
            let overlapping_pu = (0..full_set.weight().unwrap()).prop_map(move |cpu_idx| {
                topology
                    .smallest_object_covering_cpuset(&CpuSet::from(
                        full_set.iter_set().nth(cpu_idx).unwrap(),
                    ))
                    .unwrap()
            });

            // Empirically, this can pick a PU without a cpuset on weird systems
            // like GitHub's CI nodes . Address this by going up ancestors until
            // a cpuset is found.
            let overlapping_root = overlapping_pu.prop_map(|pu| {
                std::iter::once(pu)
                    .chain(pu.ancestors())
                    .find(|root| root.cpuset().is_some())
                    .unwrap()
            });

            // Insert this overlapping object at a random position
            let num_roots = disjoint_roots.len();
            (Just(disjoint_roots), overlapping_root, 0..num_roots).prop_map(
                |(mut disjoint_roots, overlapping_root, bad_root_idx)| {
                    disjoint_roots.insert(bad_root_idx, overlapping_root);
                    disjoint_roots
                },
            )
        })
    }

    proptest! {
        #[test]
        fn distribute_overlapping(
            overlapping_roots in overlapping_roots(),
            max_depth in max_depth(),
            num_items: NonZeroU8,
            flags: DistributeFlags,
        ) {
            let num_items = usize::from(num_items.get());
            prop_assert_eq!(
                Topology::test_instance().distribute_items(&overlapping_roots, num_items, max_depth, flags),
                Err(DistributeError::OverlappingRoots)
            );
        }
    }

    /// Pick a random normal object from the foreign test instance
    fn foreign_normal_object() -> impl Strategy<Value = &'static TopologyObject> {
        static FOREIGNERS: OnceLock<Box<[&'static TopologyObject]>> = OnceLock::new();
        let objects =
            FOREIGNERS.get_or_init(|| Topology::foreign_instance().normal_objects().collect());
        prop::sample::select(&objects[..])
    }

    /// To the random input provided by `disjoint_roots`, add an extra root from
    /// a different topology, and report at which index it was added
    fn foreign_roots_and_idx() -> impl Strategy<Value = (Vec<&'static TopologyObject>, usize)> {
        // Pick a set of disjoint roots and a foreign object
        (disjoint_roots(), foreign_normal_object()).prop_flat_map(
            |(disjoint_roots, foreign_object)| {
                // Insert the foreign object at a random position, report where
                let num_roots = disjoint_roots.len();
                (Just((disjoint_roots, foreign_object)), 0..num_roots).prop_map(
                    |((mut disjoint_roots, foreign_object), bad_root_idx)| {
                        disjoint_roots.insert(bad_root_idx, foreign_object);
                        (disjoint_roots, bad_root_idx)
                    },
                )
            },
        )
    }

    proptest! {
        #[test]
        fn distribute_items(
            (foreign_roots, foreign_idx) in foreign_roots_and_idx(),
            max_depth in max_depth(),
            num_items: NonZeroU8,
            flags: DistributeFlags,
        ) {
            let num_items = usize::from(num_items.get());
            let foreign_object = foreign_roots[foreign_idx];
            prop_assert_eq!(
                Topology::test_instance().distribute_items(&foreign_roots, num_items, max_depth, flags),
                Err(DistributeError::ForeignRoot(foreign_object.into()))
            );
        }
    }
}
