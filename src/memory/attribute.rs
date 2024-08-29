//! Memory attributes
//!
//! On some platforms, NUMA nodes do not represent homogeneous memory, but
//! instead reflect multiple tiers of memory with different performance
//! characteristics (HBM, DDR, NVRAM...).
//!
//! hwloc's memory attribute API, which this module is all about, lets you know
//! pick which memory you want to allocate from based on capacity, locality,
//! latency and bandwidth considerations.
//!
//! Most of this module's functionality is exposed via [methods of the Topology
//! struct](../../topology/struct.Topology.html#comparing-memory-node-attributes-for-finding-where-to-allocate-on).
//! The module itself only hosts type definitions that are related to this
//! functionality.

#[cfg(doc)]
use crate::topology::support::DiscoverySupport;
use crate::{
    bitmap::BitmapRef,
    cpu::cpuset::CpuSet,
    errors::{self, FlagsError, ForeignObjectError, HybridError, NulError, RawHwlocError},
    ffi::{
        int,
        string::LibcString,
        transparent::{AsInner, AsNewtype},
    },
    object::{TopologyObject, TopologyObjectID},
    topology::{editor::TopologyEditor, Topology},
};
use bitflags::bitflags;
use derive_more::{Display, From};
use errno::Errno;
use hwlocality_sys::{
    hwloc_const_topology_t, hwloc_local_numanode_flag_e, hwloc_location, hwloc_location_u,
    hwloc_memattr_flag_e, hwloc_memattr_id_t, hwloc_obj, HWLOC_LOCAL_NUMANODE_FLAG_ALL,
    HWLOC_LOCAL_NUMANODE_FLAG_LARGER_LOCALITY, HWLOC_LOCAL_NUMANODE_FLAG_SMALLER_LOCALITY,
    HWLOC_LOCATION_TYPE_CPUSET, HWLOC_LOCATION_TYPE_OBJECT, HWLOC_MEMATTR_FLAG_HIGHER_FIRST,
    HWLOC_MEMATTR_FLAG_LOWER_FIRST, HWLOC_MEMATTR_FLAG_NEED_INITIATOR, HWLOC_MEMATTR_ID_BANDWIDTH,
    HWLOC_MEMATTR_ID_CAPACITY, HWLOC_MEMATTR_ID_LATENCY, HWLOC_MEMATTR_ID_LOCALITY,
};
#[cfg(feature = "hwloc-2_8_0")]
use hwlocality_sys::{
    HWLOC_MEMATTR_ID_READ_BANDWIDTH, HWLOC_MEMATTR_ID_READ_LATENCY,
    HWLOC_MEMATTR_ID_WRITE_BANDWIDTH, HWLOC_MEMATTR_ID_WRITE_LATENCY,
};
use libc::{EBUSY, EINVAL, ENOENT};
#[allow(unused)]
#[cfg(test)]
use similar_asserts::assert_eq;
use std::{
    collections::HashSet,
    ffi::{c_int, c_uint, c_ulong, CStr},
    fmt::{self, Debug},
    hash::Hash,
    mem::MaybeUninit,
    ptr::{self, NonNull},
};
use thiserror::Error;

/// # Comparing memory node attributes for finding where to allocate on
///
/// Platforms with heterogeneous memory require ways to decide whether a buffer
/// should be allocated on "fast" memory (such as HBM), "normal" memory (DDR) or
/// even "slow" but large-capacity memory (non-volatile memory). These memory
/// nodes are called "Targets" while the CPU accessing them is called the
/// "Initiator". Access performance depends on their locality (NUMA platforms)
/// as well as the intrinsic performance of the targets (heterogeneous platforms).
///
/// The following attributes describe the performance of memory accesses from an
/// Initiator to a memory Target, for instance their latency or bandwidth.
/// Initiators performing these memory accesses are usually some PUs or Cores
/// (described as a CPU set). Hence a Core may choose where to allocate a memory
/// buffer by comparing the attributes of different target memory nodes nearby.
///
/// There are also some attributes that are system-wide. Their value does not
/// depend on a specific initiator performing an access. The memory node
/// capacity is an example of such attribute without initiator.
///
/// One way to use this API is to start with a cpuset describing the Cores where
/// a program is bound. The best target NUMA node for allocating memory in this
/// program on these Cores may be obtained by passing this cpuset as an
/// initiator to [`MemoryAttribute::best_target()`] with the relevant
/// memory attribute. For instance, if the code is latency limited, use the
/// Latency attribute.
///
/// A more flexible approach consists in getting the list of local NUMA nodes by
/// passing this cpuset to [`Topology::local_numa_nodes()`]. Attribute values
/// for these nodes, if any, may then be obtained with
/// [`MemoryAttribute::value()`] and manually compared with the desired criteria.
///
/// The API also supports specific objects as initiator, but it is currently not
/// used internally by hwloc. Users may for instance use it to provide custom
/// performance values for host memory accesses performed by GPUs.
/// The interface actually also accepts targets that are not NUMA nodes.
//
// --- Implementation details ---
//
// Upstream docs: https://hwloc.readthedocs.io/en/v2.9/group__hwlocality__memattrs.html
impl Topology {
    /// Identifier of the memory attribute with the given name
    ///
    /// Note that a number of predefined memory attributes have hard-coded
    /// identifiers and need not be queried by name at runtime, see the
    /// constructors of [`MemoryAttribute`] for more information.
    ///
    /// # Errors
    ///
    /// - [`NulError`] if `name` contains NUL chars.
    #[doc(alias = "hwloc_memattr_get_by_name")]
    pub fn memory_attribute_named(
        &self,
        name: &str,
    ) -> Result<Option<MemoryAttribute<'_>>, NulError> {
        let name = LibcString::new(name)?;
        let mut id = MaybeUninit::uninit();
        // SAFETY: - Topology is trusted to contain a valid ptr (type invariant)
        //         - name is trusted to be a valid C string (type invariant)
        //         - hwloc ops are trusted not to modify *const parameters
        //         - Per documentation, id is a pure out parameter that hwloc
        //           does not read
        let res = errors::call_hwloc_int_normal("hwloc_memattr_get_by_name", || unsafe {
            hwlocality_sys::hwloc_memattr_get_by_name(self.as_ptr(), name.borrow(), id.as_mut_ptr())
        });
        match res {
            // SAFETY: If hwloc indicates success, it should have initialized id
            //         to a valid memory attribute ID
            Ok(_positive) => Ok(Some(unsafe {
                MemoryAttribute::wrap(self, id.assume_init())
            })),
            Err(RawHwlocError {
                errno: Some(Errno(EINVAL)),
                ..
            }) => Ok(None),
            #[cfg(windows)]
            Err(RawHwlocError { errno: None, .. }) => {
                // As explained in the RawHwlocError documentation, errno values
                // may not correctly propagate from hwloc to hwlocality on
                // Windows. Since there is only one expected errno value here,
                // we'll interpret lack of errno as EINVAL on Windows.
                Ok(None)
            }
            Err(raw_err) => unreachable!("Unexpected hwloc error: {raw_err}"),
        }
    }

    /// Find NUMA nodes that are local to some CPUs or [`TopologyObject`].
    ///
    /// If `target` is given as a [`TopologyObject`], its CPU set is used to
    /// find NUMA nodes with the corresponding locality. If the object does not
    /// have a CPU set (e.g. I/O object), the CPU parent (where the I/O object
    /// is attached) is used.
    ///
    /// Some of these NUMA nodes may not have any memory attribute values and
    /// hence not be reported as actual targets in other functions.
    ///
    /// When an object CPU set is given as locality, for instance a Package, and
    /// when `flags` contains both [`LocalNUMANodeFlags::LARGER_LOCALITY`] and
    /// [`LocalNUMANodeFlags::SMALLER_LOCALITY`], the returned array corresponds
    /// to the nodeset of that object.
    ///
    /// # Errors
    ///
    /// - [`ForeignObjectError`] if `target` refers to a [`TopologyObject`] that
    ///   does not belong to this topology.
    #[allow(clippy::missing_errors_doc)]
    #[doc(alias = "hwloc_get_local_numanode_objs")]
    pub fn local_numa_nodes<'target>(
        &self,
        target: impl Into<NUMALocation<'target>>,
    ) -> Result<Vec<&TopologyObject>, HybridError<ForeignObjectError>> {
        /// Polymorphized version of this function (avoids generics code bloat)
        fn polymorphized<'self_>(
            self_: &'self_ Topology,
            target: NUMALocation<'_>,
        ) -> Result<Vec<&'self_ TopologyObject>, HybridError<ForeignObjectError>> {
            // Prepare to call hwloc
            // SAFETY: Will only be used before returning from this function
            let (location, flags) = unsafe { target.to_checked_raw(self_)? };
            let mut nr = 0;
            let call_ffi = |nr_mut, out_ptr| {
                // SAFETY: - Topology is trusted to contain a valid ptr (type invariant)
                //         - hwloc ops are trusted not to modify *const parameters
                //         - The NUMALocation API is designed in such a way that
                //           an invalid (location, flags) tuple cannot happen.
                //         - nr_mut and out_ptr are call site dependent, see below.
                errors::call_hwloc_int_normal("hwloc_get_local_numanode_objs", || unsafe {
                    hwlocality_sys::hwloc_get_local_numanode_objs(
                        self_.as_ptr(),
                        &location,
                        nr_mut,
                        out_ptr,
                        flags.bits(),
                    )
                })
                .map_err(HybridError::Hwloc)
            };

            // Query node count
            // SAFETY: A null output is allowed, and 0 elements is the correct way
            //         to model it in the "nr" parameter. This will set nr to the
            //         actual buffer size we need to allocate.
            call_ffi(&mut nr, ptr::null_mut())?;
            let len = int::expect_usize(nr);

            // Allocate storage and fill node list
            let mut out = vec![ptr::null(); len];
            let old_nr = nr;
            // SAFETY: out is a valid buffer of size len, which is just nr as usize
            call_ffi(&mut nr, out.as_mut_ptr())?;
            assert_eq!(old_nr, nr, "Inconsistent node count from hwloc");

            // Translate node pointers into node references
            Ok(out
                .into_iter()
                .map(|ptr| {
                    assert!(!ptr.is_null(), "Invalid NUMA node pointer from hwloc");
                    // SAFETY: We trust that if hwloc emits a non-null pointer,
                    //         it is valid, bound to the topology's lifetime,
                    //         and points to a valid target.
                    unsafe { (&*ptr).as_newtype() }
                })
                .collect())
        }
        polymorphized(self, target.into())
    }

    /// Dump the values of all memory attributes
    pub(crate) fn dump_memory_attributes(&self) -> MultiAttributeDump<'_> {
        MultiAttributeDump::new(self)
    }
}

/// Dump of memory attributes, used to implement topology Debug and comparison
#[derive(Clone)]
pub(crate) struct MultiAttributeDump<'topology>(Vec<AttributeDump<'topology>>);
//
impl<'topology> MultiAttributeDump<'topology> {
    /// Dump all memory attributes
    fn new(topology: &'topology Topology) -> Self {
        Self(
            MemoryAttribute::all(topology)
                .map(AttributeDump::new)
                .collect(),
        )
    }

    /// Truth that this dump contains the same data as another dump, assuming
    /// both dumps originate from related topologies.
    ///
    /// By related, we mean that `other` should either originate from the same
    /// [`Topology`] as `self`, or from a (possibly modified) clone of that
    /// topology, which allows us to use object global persistent indices as
    /// object identifiers.
    ///
    /// Comparing dumps from unrelated topologies will yield an unpredictable
    /// boolean value.
    pub(crate) fn eq_modulo_topology(&self, other: &Self) -> bool {
        if self.0.len() != other.0.len() {
            return false;
        }
        self.0
            .iter()
            .zip(&other.0)
            .all(|(dump1, dump2)| dump1.eq_modulo_topology(dump2))
    }
}
//
impl Debug for MultiAttributeDump<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut attr_to_dump = f.debug_map();
        for attr_dump in &self.0 {
            attr_to_dump.entry(&attr_dump.attribute, &attr_dump.targets);
        }
        attr_to_dump.finish()
    }
}

/// Dump of a specific memory attribute
#[derive(Clone, Debug)]
struct AttributeDump<'topology> {
    /// Attribute being dumped
    attribute: MemoryAttribute<'topology>,

    /// Data dump for each known attribute target
    targets: AttributeDumpTargets<'topology>,
}
//
impl<'topology> AttributeDump<'topology> {
    /// Dump the value of the attribute for each registered target
    fn new(attribute: MemoryAttribute<'topology>) -> Self {
        Self {
            attribute,
            targets: AttributeDumpTargets::new(attribute),
        }
    }

    /// Truth that this dump contains the same data as another dump, assuming
    /// both dumps originate from related topologies.
    ///
    /// By related, we mean that `other` should either originate from the same
    /// [`Topology`] as `self`, or from a (possibly modified) clone of that
    /// topology, which allows us to use object global persistent indices as
    /// object identifiers.
    ///
    /// Comparing dumps from unrelated topologies will yield an unpredictable
    /// boolean value.
    fn eq_modulo_topology(&self, other: &Self) -> bool {
        if self.attribute.id != other.attribute.id {
            return false;
        }
        self.targets.eq_modulo_topology(&other.targets)
    }
}

/// `targets` field of `AttributeDump`
///
/// Needs to be its own struct due to design limitations of the
/// `Debug`/`Formatter` machinery.
#[derive(Clone)]
struct AttributeDumpTargets<'topology>(Vec<TargetAttributeDump<'topology>>);
//
impl<'topology> AttributeDumpTargets<'topology> {
    /// Dump the value of the attribute for each target on the system
    fn new(attribute: MemoryAttribute<'topology>) -> Self {
        let (targets, _values) = attribute
            .targets(None)
            .expect("targets() should never panic with a None input");
        Self(
            targets
                .into_iter()
                .map(|target| TargetAttributeDump::new(attribute, target))
                .collect(),
        )
    }

    /// Truth that this dump contains the same data as another dump, assuming
    /// both dumps originate from related topologies.
    ///
    /// By related, we mean that `other` should either originate from the same
    /// [`Topology`] as `self`, or from a (possibly modified) clone of that
    /// topology, which allows us to use object global persistent indices as
    /// object identifiers.
    ///
    /// Comparing dumps from unrelated topologies will yield an unpredictable
    /// boolean value.
    fn eq_modulo_topology(&self, other: &Self) -> bool {
        if self.0.len() != other.0.len() {
            return false;
        }
        self.0
            .iter()
            .zip(&other.0)
            .all(|(dump1, dump2)| dump1.eq_modulo_topology(dump2))
    }
}
//
impl Debug for AttributeDumpTargets<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut targets_to_initiators = f.debug_map();
        for target_dump in &self.0 {
            targets_to_initiators.entry(&target_dump.target, &target_dump.initiators_and_values);
        }
        targets_to_initiators.finish()
    }
}

/// Dump of a specific memory attribute for a specific target
#[derive(Clone, Debug)]
struct TargetAttributeDump<'topology> {
    /// Target for which the memory attribute values were queried
    target: &'topology TopologyObject,

    /// Result of the memory attribute value query
    initiators_and_values: InitiatorsAndValues<'topology>,
}
//
impl<'topology> TargetAttributeDump<'topology> {
    /// Dump the value of the attribute for a specific target
    ///
    /// # Panics
    ///
    /// Expects `target` to belong to the same topology as `attribute`.
    fn new(attribute: MemoryAttribute<'topology>, target: &'topology TopologyObject) -> Self {
        Self {
            target,
            initiators_and_values: InitiatorsAndValues::new(attribute, target),
        }
    }

    /// Truth that this dump contains the same data as another dump, assuming
    /// both dumps originate from related topologies.
    ///
    /// By related, we mean that `other` should either originate from the same
    /// [`Topology`] as `self`, or from a (possibly modified) clone of that
    /// topology, which allows us to use object global persistent indices as
    /// object identifiers.
    ///
    /// Comparing dumps from unrelated topologies will yield an unpredictable
    /// boolean value.
    pub(crate) fn eq_modulo_topology(&self, other: &Self) -> bool {
        if self.target.global_persistent_index() != other.target.global_persistent_index() {
            return false;
        }
        self.initiators_and_values
            .eq_modulo_topology(&other.initiators_and_values)
    }
}

/// `initiators_and_values` field of `TargetAttributeDump`
///
/// Needs to be its own struct due to design limitations of the
/// `Debug`/`Formatter` machinery.
#[derive(Clone)]
struct InitiatorsAndValues<'topology> {
    /// Initiators for which attribute values were recorded, if any
    initiators: Option<Vec<MemoryAttributeLocation<'topology>>>,

    /// Recorded attribute value (only 1 if no initiator)
    values: Vec<u64>,
}
//
impl<'topology> InitiatorsAndValues<'topology> {
    /// Dump the value of the attribute for a specific target
    ///
    /// # Panics
    ///
    /// Expects `target` to belong to the same topology as `attribute`.
    fn new(attribute: MemoryAttribute<'topology>, target: &'topology TopologyObject) -> Self {
        if attribute
            .flags()
            .contains(MemoryAttributeFlags::NEED_INITIATOR)
        {
            attribute
                .initiators(target)
                .map(|(initiators, values)| Self {
                    initiators: Some(initiators),
                    values,
                })
                .expect(
                    "initiator provided when NEED_INITIATOR, \
                    target is local, no other known error case",
                )
        } else {
            attribute
                .value(None, target)
                .map(|value| Self {
                    initiators: None,
                    values: value.map_or_else(Vec::new, |value| vec![value]),
                })
                .expect(
                    "initiator not provided in absence of NEED_INITIATOR, \
                    target is local, no other known error case",
                )
        }
    }

    /// Truth that this dump contains the same data as another dump, assuming
    /// both dumps originate from related topologies.
    ///
    /// By related, we mean that `other` should either originate from the same
    /// [`Topology`] as `self`, or from a (possibly modified) clone of that
    /// topology, which allows us to use object global persistent indices as
    /// object identifiers.
    ///
    /// Comparing dumps from unrelated topologies will yield an unpredictable
    /// boolean value.
    pub(crate) fn eq_modulo_topology(&self, other: &Self) -> bool {
        if self.values != other.values {
            return false;
        }
        let (initiators1, initiators2) = match (&self.initiators, &other.initiators) {
            (Some(i1), Some(i2)) => (i1, i2),
            (None, None) => return true,
            (Some(_), None) | (None, Some(_)) => return false,
        };
        if initiators1.len() != initiators2.len() {
            return false;
        }
        initiators1.iter().zip(initiators2.iter()).all(|(i1, i2)| {
            use MemoryAttributeLocation as MAL;
            match (i1, i2) {
                (MAL::CpuSet(set1), MAL::CpuSet(set2)) => set1 == set2,
                (MAL::Object(obj1), MAL::Object(obj2)) => {
                    obj1.global_persistent_index() == obj2.global_persistent_index()
                }
                (MAL::CpuSet(_), MAL::Object(_)) | (MAL::Object(_), MAL::CpuSet(_)) => false,
            }
        })
    }
}
//
impl Debug for InitiatorsAndValues<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Handle initiator-less attributes
        let Some(initiators) = &self.initiators else {
            assert_eq!(
                self.values.len(),
                1,
                "memory attributes without an initiator should have only one value"
            );
            return write!(f, "{}", self.values[0]);
        };

        // Handle initiator-ful attributes
        let mut initiator_to_value = f.debug_map();
        for (initiator, value) in initiators.iter().zip(&self.values) {
            initiator_to_value.entry(&initiator, &value);
        }
        initiator_to_value.finish()
    }
}

/// # Managing memory attributes
//
// --- Implementation details ---
//
// Upstream docs: https://hwloc.readthedocs.io/en/v2.9/group__hwlocality__memattrs__manage.html
impl<'topology> TopologyEditor<'topology> {
    /// Register a new memory attribute
    ///
    /// # Errors
    ///
    /// - [`BadFlags`] if `flags` does not contain exactly one of the
    ///   [`HIGHER_IS_BEST`] and [`LOWER_IS_BEST`] flags.
    /// - [`NameContainsNul`] if `name` contains NUL chars.
    /// - [`NameTaken`] if another attribute called `name` already exists.
    ///
    /// [`BadFlags`]: RegisterError::BadFlags
    /// [`HIGHER_IS_BEST`]: [`MemoryAttributeFlags::HIGHER_IS_BEST`]
    /// [`LOWER_IS_BEST`]: [`MemoryAttributeFlags::LOWER_IS_BEST`]
    /// [`NameContainsNul`]: RegisterError::NameContainsNul
    /// [`NameTaken`]: RegisterError::NameTaken
    #[doc(alias = "hwloc_memattr_register")]
    pub fn register_memory_attribute(
        &mut self,
        name: &str,
        flags: MemoryAttributeFlags,
    ) -> Result<MemoryAttributeBuilder<'_, 'topology>, RegisterError> {
        if !flags.is_valid() {
            return Err(flags.into());
        }
        let libc_name = LibcString::new(name)?;
        let mut id = hwloc_memattr_id_t::MAX;
        // SAFETY: - Topology is trusted to contain a valid ptr (type invariant)
        //         - libc_name is trusted to be a valid C string (type
        //           invariant)
        //         - hwloc ops are trusted not to modify *const parameters
        //         - hwloc ops are trusted to keep *mut parameters in a
        //           valid state unless stated otherwise
        //         - flags are validated to be correct
        //         - id is an out-parameter, so it can take any input value
        let res = errors::call_hwloc_int_normal("hwloc_memattr_register", || unsafe {
            hwlocality_sys::hwloc_memattr_register(
                self.topology_mut_ptr(),
                libc_name.borrow(),
                flags.bits(),
                &mut id,
            )
        });
        let handle_ebusy = || Err(RegisterError::NameTaken(name.into()));
        match res {
            Ok(_positive) => Ok(MemoryAttributeBuilder {
                editor: self,
                flags,
                id,
                name: name.into(),
            }),
            Err(RawHwlocError {
                errno: Some(Errno(EBUSY)),
                ..
            }) => handle_ebusy(),
            #[cfg(windows)]
            Err(RawHwlocError { errno: None, .. }) => {
                // As explained in the RawHwlocError documentation, errno values
                // may not correctly propagate from hwloc to hwlocality on
                // Windows. Since there is only one expected errno value here,
                // we'll interpret lack of errno as EBUSY on Windows.
                handle_ebusy()
            }
            Err(raw_err) => unreachable!("Unexpected hwloc error: {raw_err}"),
        }
    }
}

/// Error returned when trying to create an memory attribute
#[cfg_attr(windows, allow(variant_size_differences))]
#[derive(Clone, Debug, Eq, Error, Hash, PartialEq)]
pub enum RegisterError {
    /// Provided `name` contains NUL chars
    #[error("memory attribute name can't contain NUL chars")]
    NameContainsNul,

    /// Another attribute already uses this name
    #[error("there is already a memory attribute named \"{0}\"")]
    NameTaken(Box<str>),

    /// Specified flags are not correct
    ///
    /// You must specify exactly one of the [`HIGHER_IS_BEST`] and
    /// [`LOWER_IS_BEST`] flags.
    ///
    /// [`HIGHER_IS_BEST`]: [`MemoryAttributeFlags::HIGHER_IS_BEST`]
    /// [`LOWER_IS_BEST`]: [`MemoryAttributeFlags::LOWER_IS_BEST`]
    #[error(transparent)]
    BadFlags(#[from] FlagsError<MemoryAttributeFlags>),
}
//
impl From<NulError> for RegisterError {
    fn from(_: NulError) -> Self {
        Self::NameContainsNul
    }
}
//
impl From<MemoryAttributeFlags> for RegisterError {
    fn from(value: MemoryAttributeFlags) -> Self {
        Self::BadFlags(value.into())
    }
}

/// Mechanism to configure a memory attribute
//
// --- Implementation details ---
//
// # Safety
//
// `id` must be a valid new memory attribute ID from `hwloc_memattr_register()`
#[derive(Debug)]
pub struct MemoryAttributeBuilder<'editor, 'topology> {
    /// Underlying [`TopologyEditor`]
    editor: &'editor mut TopologyEditor<'topology>,

    /// Flags which this memory attribute was registered with
    flags: MemoryAttributeFlags,

    /// ID that `hwloc_memattr_register` allocated to this memory attribute
    id: hwloc_memattr_id_t,

    /// Name of this memory attribute
    name: Box<str>,
}
//
impl MemoryAttributeBuilder<'_, '_> {
    /// Set attribute values for (initiator, target node) pairs
    ///
    /// Given a read-only view of the underlying [`Topology`], the provided
    /// `find_values` callback should conceptually extract a list of
    /// `(initiator, target, value)` tuples if this memory attribute has
    /// initiators (flag [`MemoryAttributeFlags::NEED_INITIATOR`] was set at
    /// attribute registration time), and a list of `(target, value)` tuples
    /// if the attribute has no initiators.
    ///
    /// However, for efficiency reasons, the callback does not literally return
    /// a list of ternary tuples with an optional initiator member, as this
    /// would require one initiator check per attribute value. Instead, the
    /// callback returns a list of `(target, value)` pairs along with an
    /// optional list of initiators. If a list of initiators is provided, it
    /// must be as long as the provided list of `(target, value)` pairs.
    ///
    /// Initiators should be specified as
    /// [`CpuSet`](MemoryAttributeLocation::CpuSet) when referring to accesses
    /// performed by CPU cores. The [`Object`](MemoryAttributeLocation::Object)
    /// initiator type is currently  unused internally by hwloc, but users may
    /// for instance use it to provide custom information about host memory
    /// accesses performed by GPUs.
    ///
    /// # Errors
    ///
    /// - [`InconsistentDataLen`] if the number of provided initiators and
    ///   attribute values does not match
    /// - [`MultipleValues`] if multiple values are provided for the same
    ///   (initiator, target) pair
    /// - [`ForeignInitiator`] if some of the provided initiators are
    ///   [`TopologyObject`]s that do not belong to this [`Topology`]
    /// - [`ForeignTarget`] if some of the provided targets are
    ///   [`TopologyObject`]s that do not belong to this [`Topology`]
    /// - [`NeedInitiator`] if initiators were not specified for an attribute
    ///   that requires them
    /// - [`UnwantedInitiator`] if initiators were specified for an attribute
    ///   that does not requires them
    ///
    /// [`ForeignInitiator`]: InitiatorInputError::ForeignInitiator
    /// [`ForeignTarget`]: ValueInputError::ForeignTarget
    /// [`InconsistentDataLen`]: ValueInputError::InconsistentDataLen
    /// [`MultipleValues`]: ValueInputError::MultipleValues
    /// [`NeedInitiator`]: InitiatorInputError::NeedInitiator
    /// [`UnwantedInitiator`]: InitiatorInputError::UnwantedInitiator
    #[doc(alias = "hwloc_memattr_set_value")]
    pub fn set_values(
        &mut self,
        find_values: impl FnOnce(
            &Topology,
        ) -> (
            Option<Vec<MemoryAttributeLocation<'_>>>,
            Vec<(&TopologyObject, u64)>,
        ),
    ) -> Result<(), HybridError<ValueInputError>> {
        /// Polymorphized subset of this function (reduces generics code bloat)
        ///
        /// # Safety
        ///
        /// - `initiators` must have just gone through the `to_checked_raw()`
        ///   validation process against this attribute's topology.
        /// - `target_ptrs_and_values` must only contain pointers to objects
        ///   from this memory attribute's topology.
        unsafe fn polymorphized(
            self_: &mut MemoryAttributeBuilder<'_, '_>,
            initiators: RawInitiators,
            target_ptrs_and_values: RawTargetsAndValues,
        ) -> Result<(), HybridError<ValueInputError>> {
            // Massage initiators into an iterator of what hwloc wants
            let initiator_ptrs = initiators
                .iter()
                .flatten()
                .map(|initiator_ref| {
                    let initiator_ptr: *const hwloc_location = initiator_ref;
                    initiator_ptr
                })
                .chain(std::iter::repeat_with(ptr::null));

            // Set memory attribute values
            for (initiator_ptr, (target_ptr, value)) in initiator_ptrs.zip(target_ptrs_and_values) {
                // SAFETY: - Topology is trusted to contain a valid ptr (type invariant)
                //         - hwloc ops are trusted to keep *mut parameters in a
                //           valid state unless stated otherwise
                //         - id from hwloc_memattr_register is trusted to be valid
                //         - target_ptr was checked to belong to this topology
                //         - initiator_ptr was checked to belong to this topology
                //         - Flags must be 0 for now
                //         - Any attribute value is valid
                errors::call_hwloc_int_normal("hwloc_memattr_set_value", || unsafe {
                    hwlocality_sys::hwloc_memattr_set_value(
                        self_.editor.topology_mut_ptr(),
                        self_.id,
                        target_ptr.as_ptr(),
                        initiator_ptr,
                        0,
                        value,
                    )
                })
                .map_err(HybridError::Hwloc)?;
            }
            Ok(())
        }

        // Run user callback
        let topology = self.editor.topology();
        let (initiators, targets_and_values) = find_values(topology);

        // Check user inputs and get them closer to hwloc expectations
        //
        // SAFETY: Raw initiators and targets will be used before the end of
        //         this function, without tampering
        let (initiators, target_ptrs_and_values) = unsafe {
            Self::checked_set_values_inputs(
                self.flags,
                &self.name,
                topology,
                initiators.as_deref(),
                &targets_and_values[..],
            )?
        };

        // Invoke polymorphized subset with the results
        // SAFETY: Initiators and target_ptrs_and_values have been checked to
        //         belong to this memory attribute's topology, and initiators
        //         have not been tampered with.
        unsafe { polymorphized(self, initiators, target_ptrs_and_values) }
    }

    /// Check inputs to `set_values` for usage errors. If everything is alright,
    /// return lower-level inputs for the polymorphic subset of `set_values`.
    ///
    /// # Safety
    ///
    /// The output values must be used before the end of the lifetime of the
    /// `initiators` and `targets_and_values` inputs. Inputs should not have
    /// been tampered with before this happens.
    unsafe fn checked_set_values_inputs<'topology>(
        flags: MemoryAttributeFlags,
        name: &str,
        topology: &'topology Topology,
        initiators: Option<&[MemoryAttributeLocation<'topology>]>,
        targets_and_values: &[(&'topology TopologyObject, u64)],
    ) -> Result<(RawInitiators, RawTargetsAndValues), HybridError<ValueInputError>> {
        // Set up output storage
        let mut out_initiators = None;
        let mut target_ptrs_and_values = Vec::with_capacity(targets_and_values.len());

        /// Common code for adding a (target, value) pair + checking the target
        macro_rules! try_push_target_and_value {
            ($target_and_value:expr) => {
                let (target_ref, value): (&TopologyObject, u64) = $target_and_value;
                if !topology.contains(target_ref) {
                    return Err(ValueInputError::ForeignTarget(target_ref.into()).into());
                }
                let target_ptr = NonNull::from(target_ref).as_inner();
                target_ptrs_and_values.push((target_ptr, value));
            };
        }

        // Case where the user provided initiators
        if let Some(initiators) = &initiators {
            // The memory attribute should call for an initiator
            if !flags.contains(MemoryAttributeFlags::NEED_INITIATOR) {
                return Err(ValueInputError::BadInitiators(
                    InitiatorInputError::UnwantedInitiator(name.into()),
                )
                .into());
            }

            // There should be as many initiators as there are targets/values
            if initiators.len() != targets_and_values.len() {
                return Err(ValueInputError::InconsistentDataLen.into());
            }

            /// Simplified [`MemoryAttributeLocation`] that fits in [`HashSet`]
            #[allow(clippy::missing_docs_in_private_items)]
            #[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
            enum InitiatorId<'a> {
                CpuSet(BitmapRef<'a, CpuSet>),
                Object(TopologyObjectID),
            }

            // Iterate over (initiator, target, value) triplets
            let mut unique_initiator_targets = HashSet::with_capacity(initiators.len());
            let out_initiators = out_initiators.insert(Vec::with_capacity(initiators.len()));
            for (initiator, (target, value)) in
                initiators.iter().zip(targets_and_values.iter().copied())
            {
                // Check for (initiator, target) uniqueness
                let initiator_id = match initiator {
                    MemoryAttributeLocation::CpuSet(set) => InitiatorId::CpuSet(*set),
                    MemoryAttributeLocation::Object(obj) => {
                        InitiatorId::Object(obj.global_persistent_index())
                    }
                };
                if !unique_initiator_targets
                    .insert((initiator_id, target.global_persistent_index()))
                {
                    return Err(ValueInputError::MultipleValues.into());
                }

                // Record initiator, target and value into output Vecs
                out_initiators.push(
                    // SAFETY: Per function precondition
                    unsafe {
                        initiator.to_checked_raw(topology).map_err(|e| {
                            ValueInputError::BadInitiators(InitiatorInputError::ForeignInitiator(e))
                        })?
                    },
                );
                try_push_target_and_value!((target, value));
            }
        } else {
            // Case where the user did not provide initiators. This is only
            // valid if the memory attribute doesn't call for initiators
            if flags.contains(MemoryAttributeFlags::NEED_INITIATOR) {
                return Err(
                    ValueInputError::BadInitiators(InitiatorInputError::NeedInitiator(name.into()))
                        .into(),
                );
            }

            // Record (target, value) pairs, tracking target uniqueness
            let mut unique_target_ids = HashSet::with_capacity(targets_and_values.len());
            for (target, value) in targets_and_values.iter().copied() {
                if !unique_target_ids.insert(target.global_persistent_index()) {
                    return Err(ValueInputError::MultipleValues.into());
                }
                try_push_target_and_value!((target, value));
            }
        }
        Ok((out_initiators, target_ptrs_and_values))
    }
}

/// Predigested initiators for [`MemoryAttributeBuilder::set_values`]
type RawInitiators = Option<Vec<hwloc_location>>;

/// Predigested targets and values for [`MemoryAttributeBuilder::set_values`]
type RawTargetsAndValues = Vec<(NonNull<hwloc_obj>, u64)>;

/// Error returned by [`MemoryAttributeBuilder::set_values`] when the
/// `find_values` callback returns incorrect initiators or targets
#[derive(Clone, Debug, Eq, Error, Hash, PartialEq)]
pub enum ValueInputError {
    /// The number of provided initiators does not match the number of attribute values
    #[error("number of memory attribute values doesn't match number of initiators")]
    InconsistentDataLen,

    /// Multiple values are provided for the same (initiator, target) pair
    #[error("multiple values provided for the same (initiator, target) pair")]
    MultipleValues,

    /// Specified initiators for these attribute values are not valid
    #[error(transparent)]
    BadInitiators(#[from] InitiatorInputError),

    /// Some provided targets are [`TopologyObject`]s that do not belong to
    /// the topology being modified
    #[error("memory attribute target {0}")]
    ForeignTarget(ForeignObjectError),
}

/// Generate [`MemoryAttribute`] constructors around predefined memory attribute
/// IDs from hwloc with minimal boilerplate
///
/// # Safety
///
/// IDs must be ID constants from hwloc
macro_rules! wrap_ids_unchecked {
    (
        $(
            $(#[$attr:meta])*
            $id:ident -> $constructor:ident
        ),*
    ) => {
        $(
            $(#[$attr])*
            // FIXME: Not supported by rustdoc yet, see
            //        https://github.com/rust-lang/rust/issues/94180
            //
            // #[doc(alias = stringify!($id))]
            pub const fn $constructor(topology: &'topology Topology) -> Self {
                // SAFETY: Per macro precondition
                unsafe { Self::wrap(topology, $id) }
            }
        )*
    };
}

/// Memory attribute
///
/// May be either one of the predefined attributes (see associated const fns)
/// or a new attribute created using
/// [`TopologyEditor::register_memory_attribute()`].
//
// --- Implementation details ---
//
// # Safety
//
// `id` must be a valid identifier to a memory attribute known of `topology`.
#[derive(Copy, Clone)]
#[doc(alias = "hwloc_memattr_id_e")]
#[doc(alias = "hwloc_memattr_id_t")]
pub struct MemoryAttribute<'topology> {
    /// Topology for which memory attribute is defined
    topology: &'topology Topology,

    /// Identifier of the memory attribute being manipulated
    id: hwloc_memattr_id_t,
}
//
/// # Predefined memory attributes
impl<'topology> MemoryAttribute<'topology> {
    wrap_ids_unchecked!(
        /// Node capacity in bytes (see [`TopologyObject::total_memory()`])
        ///
        /// This attribute involves no initiator.
        ///
        /// Requires [`DiscoverySupport::numa_memory()`].
        HWLOC_MEMATTR_ID_CAPACITY -> capacity,

        /// Number of PUs in that locality (i.e. cpuset weight)
        ///
        /// Smaller locality is better. This attribute involves no initiator.
        ///
        /// Requires [`DiscoverySupport::pu_count()`].
        HWLOC_MEMATTR_ID_LOCALITY -> locality,

        /// Average bandwidth in MiB/s, as seen from the given initiator location
        ///
        /// This is the average bandwidth for read and write accesses. If the
        /// platform provides individual read and write bandwidths but no
        /// explicit average value, hwloc computes and returns the average.
        HWLOC_MEMATTR_ID_BANDWIDTH -> bandwidth,

        /// Read bandwidth in MiB/s, as seen from the given initiator location
        #[cfg(feature = "hwloc-2_8_0")]
        HWLOC_MEMATTR_ID_READ_BANDWIDTH -> read_bandwidth,

        /// Write bandwidth in MiB/s, as seen from the given initiator location
        #[cfg(feature = "hwloc-2_8_0")]
        HWLOC_MEMATTR_ID_WRITE_BANDWIDTH -> write_bandwidth,

        /// Average latency in nanoseconds, as seen from the given initiator location
        ///
        /// This is the average latency for read and write accesses. If the
        /// platform value provides individual read and write latencies but no
        /// explicit average, hwloc computes and returns the average.
        HWLOC_MEMATTR_ID_LATENCY -> latency,

        /// Read latency in nanoseconds, as seen from the given initiator location
        #[cfg(feature = "hwloc-2_8_0")]
        HWLOC_MEMATTR_ID_READ_LATENCY -> read_latency,

        /// Write latency in nanoseconds, as seen from the given initiator location
        #[cfg(feature = "hwloc-2_8_0")]
        HWLOC_MEMATTR_ID_WRITE_LATENCY -> write_latency

        // TODO: If you add new attributes, add support to static_flags and
        //       a matching MemoryAttribute constructor below
    );

    /// Enumerate all registered memory attributes
    fn all(topology: &'topology Topology) -> impl Iterator<Item = Self> {
        (0..).map_while(move |id| {
            let mut flags = 0;
            // SAFETY: - Topology is trusted to contain a valid ptr (type invariant)
            //         - hwloc ops are trusted not to modify *const parameters
            //         - It has been stated by upstream that probing flags for
            //           an invalid ID is fine and will just lead to EINVAL
            //         - flags is an out parameter, its initial value doesn't matter
            let res = errors::call_hwloc_int_normal("hwloc_memattr_get_flags", || unsafe {
                hwlocality_sys::hwloc_memattr_get_flags(topology.as_ptr(), id, &mut flags)
            });
            match res {
                Ok(_positive) => Some(
                    // SAFETY: Checked id validity per upstream suggestion
                    unsafe { Self::wrap(topology, id) },
                ),
                Err(RawHwlocError {
                    errno: Some(Errno(EINVAL)),
                    ..
                }) => None,
                #[cfg(windows)]
                Err(RawHwlocError { errno: None, .. }) => {
                    // As explained in the RawHwlocError documentation, errno values
                    // may not correctly propagate from hwloc to hwlocality on
                    // Windows. Since there is only one expected errno value here,
                    // we'll interpret lack of errno as EINVAL on Windows.
                    None
                }
                Err(raw_err) => unreachable!("Unexpected hwloc error: {raw_err}"),
            }
        })
    }
}
//
/// # Memory attribute API
impl<'topology> MemoryAttribute<'topology> {
    /// Extend an [`hwloc_memattr_id_t`] with topology access to enable method syntax
    ///
    /// # Safety
    ///
    /// `id` must be a valid memory attribute ID, corresponding either to one of
    /// hwloc's predefined attributes or an attribute that was user-allocated
    /// using `hwloc_memattr_register()`.
    pub(crate) const unsafe fn wrap(topology: &'topology Topology, id: hwloc_memattr_id_t) -> Self {
        Self { id, topology }
    }

    /// Name of this memory attribute
    #[doc(alias = "hwloc_memattr_get_name")]
    pub fn name(&self) -> &'topology CStr {
        let mut name = ptr::null();
        // SAFETY: - Topology is trusted to contain a valid ptr (type invariant)
        //         - hwloc ops are trusted not to modify *const parameters
        //         - id is assumed to be valid (type invariant)
        //         - name is an out parameter, its initial value doesn't matter
        let res = errors::call_hwloc_int_normal("hwloc_memattr_get_name", || unsafe {
            hwlocality_sys::hwloc_memattr_get_name(self.topology.as_ptr(), self.id, &mut name)
        });
        let handle_einval =
            || unreachable!("MemoryAttribute should only hold valid attribute indices");
        match res {
            Ok(_positive) => {}
            Err(RawHwlocError {
                errno: Some(Errno(EINVAL)),
                ..
            }) => handle_einval(),
            #[cfg(windows)]
            Err(RawHwlocError { errno: None, .. }) => {
                // As explained in the RawHwlocError documentation, errno values
                // may not correctly propagate from hwloc to hwlocality on
                // Windows. Since there is only one expected errno value here,
                // we'll interpret lack of errno as EINVAL on Windows.
                handle_einval()
            }
            Err(raw_err) => unreachable!("Unexpected hwloc error: {raw_err}"),
        }
        assert!(
            !name.is_null(),
            "Memory attributes should have non-NULL names"
        );
        // SAFETY: If hwloc does emit a non-NULL name in a successful query run,
        //         we trust that name to be a valid `char*` C string pointer
        //         bound to the topology's lifetime
        unsafe { CStr::from_ptr(name) }
    }

    /// Flags of this memory attribute
    #[doc(alias = "hwloc_memattr_get_flags")]
    pub fn flags(&self) -> MemoryAttributeFlags {
        let flags = Self::static_flags(self.id).unwrap_or_else(|| self.dynamic_flags());
        assert!(flags.is_valid(), "hwloc emitted invalid flags");
        flags
    }

    /// Tell attribute flags if known at compile time
    fn static_flags(id: hwloc_memattr_id_t) -> Option<MemoryAttributeFlags> {
        let bandwidth_flags =
            Some(MemoryAttributeFlags::HIGHER_IS_BEST | MemoryAttributeFlags::NEED_INITIATOR);
        let latency_flags =
            Some(MemoryAttributeFlags::LOWER_IS_BEST | MemoryAttributeFlags::NEED_INITIATOR);
        match id {
            HWLOC_MEMATTR_ID_CAPACITY => Some(MemoryAttributeFlags::HIGHER_IS_BEST),
            HWLOC_MEMATTR_ID_LOCALITY => Some(MemoryAttributeFlags::LOWER_IS_BEST),
            HWLOC_MEMATTR_ID_BANDWIDTH => bandwidth_flags,
            #[cfg(feature = "hwloc-2_8_0")]
            HWLOC_MEMATTR_ID_READ_BANDWIDTH | HWLOC_MEMATTR_ID_WRITE_BANDWIDTH => bandwidth_flags,
            HWLOC_MEMATTR_ID_LATENCY => latency_flags,
            #[cfg(feature = "hwloc-2_8_0")]
            HWLOC_MEMATTR_ID_READ_LATENCY | HWLOC_MEMATTR_ID_WRITE_LATENCY => latency_flags,
            _ => None,
        }
    }

    /// Dynamically query this memory attribute's flags
    fn dynamic_flags(&self) -> MemoryAttributeFlags {
        let mut flags = 0;
        // SAFETY: - Topology is trusted to contain a valid ptr (type invariant)
        //         - hwloc ops are trusted not to modify *const parameters
        //         - id is assumed to be valid (type invariant)
        //         - flags is an out parameter, its initial value doesn't matter
        let res = errors::call_hwloc_int_normal("hwloc_memattr_get_flags", || unsafe {
            hwlocality_sys::hwloc_memattr_get_flags(self.topology.as_ptr(), self.id, &mut flags)
        });
        let handle_einval =
            || unreachable!("MemoryAttribute should only hold valid attribute indices");
        match res {
            Ok(_positive) => MemoryAttributeFlags::from_bits_truncate(flags),
            Err(RawHwlocError {
                errno: Some(Errno(EINVAL)),
                ..
            }) => handle_einval(),
            #[cfg(windows)]
            Err(RawHwlocError { errno: None, .. }) => {
                // As explained in the RawHwlocError documentation, errno values
                // may not correctly propagate from hwloc to hwlocality on
                // Windows. Since there is only one expected errno value here,
                // we'll interpret lack of errno as EINVAL on Windows.
                handle_einval()
            }
            Err(raw_err) => unreachable!("Unexpected hwloc error: {raw_err}"),
        }
    }

    /// Value of this attribute for a specific initiator and target NUMA node,
    /// if any.
    ///
    /// `initiator` should be specified if and only if this attribute has the
    /// flag [`MemoryAttributeFlags::NEED_INITIATOR`].
    ///
    /// The initiator should be a [`CpuSet`] when refering to accesses performed
    /// by CPU cores. [`MemoryAttributeLocation::Object`] is currently unused
    /// internally by hwloc, but user-defined memory attributes may for instance
    /// use it to provide custom information about host memory accesses
    /// performed by GPUs.
    ///
    /// # Errors
    ///
    /// - [`ForeignInitiator`] if the `initiator` parameter was set to a
    ///   [`TopologyObject`] that does not belong to this topology
    /// - [`ForeignTarget`] if the `target_node` object does not belong to this
    ///   topology
    /// - [`NeedInitiator`] if no `initiator` was provided but this memory
    ///   attribute needs one
    /// - [`UnwantedInitiator`] if an `initiator` was provided but this memory
    ///   attribute doesn't need one
    ///
    /// [`ForeignInitiator`]: InitiatorInputError::ForeignInitiator
    /// [`ForeignTarget`]: ValueQueryError::ForeignTarget
    /// [`NeedInitiator`]: InitiatorInputError::NeedInitiator
    /// [`UnwantedInitiator`]: InitiatorInputError::UnwantedInitiator
    #[doc(alias = "hwloc_memattr_get_value")]
    pub fn value(
        &self,
        initiator: Option<MemoryAttributeLocation<'_>>,
        target_node: &TopologyObject,
    ) -> Result<Option<u64>, HybridError<ValueQueryError>> {
        // Check and translate initiator argument
        // SAFETY: Will only be used before returning from this function
        let initiator = unsafe {
            self.checked_initiator(initiator, false)
                .map_err(|err| HybridError::Rust(ValueQueryError::BadInitiator(err)))?
        };

        // Check target argument
        if !self.topology.contains(target_node) {
            return Err(ValueQueryError::ForeignTarget(target_node.into()).into());
        }

        // Run the query
        let mut value = u64::MAX;
        // SAFETY: - Topology is trusted to contain a valid ptr (type invariant)
        //         - hwloc ops are trusted not to modify *const parameters
        //         - id is trusted to be valid (type invariant)
        //         - target_node has been checked to come from this topology
        //         - initiator has been checked to come from this topology and
        //           to be NULL if and only if the attribute has no initiator
        //         - flags must be 0
        //         - Value is an out parameter, its initial value isn't read
        let res = errors::call_hwloc_int_normal("hwloc_memattr_get_value", || unsafe {
            hwlocality_sys::hwloc_memattr_get_value(
                self.topology.as_ptr(),
                self.id,
                target_node.as_inner(),
                &initiator,
                0,
                &mut value,
            )
        });
        match res {
            Ok(_) => Ok(Some(value)),
            // We should have handled all sources of EINVAL besides calling
            // value() with an (initiator, target_node) for which there is no
            // memory attribute value, so returning None feels appropriate here.
            Err(RawHwlocError {
                errno: Some(Errno(EINVAL)),
                ..
            }) => Ok(None),
            #[cfg(windows)]
            Err(RawHwlocError { errno: None, .. }) => {
                // As explained in the RawHwlocError documentation, errno values
                // may not correctly propagate from hwloc to hwlocality on
                // Windows. But only EINVAL is expected here so we're fine.
                Ok(None)
            }
            Err(other_err) => Err(HybridError::Hwloc(other_err)),
        }
    }

    /// Best target node and associated attribute value, if any, for a given initiator
    ///
    /// The notes on initiator semantics in [`MemoryAttribute::value()`] also
    /// apply to this function.
    ///
    /// If multiple targets have the same attribute values, only one is returned
    /// (and there is no way to clarify how that one is chosen). Applications
    /// that want to detect targets with identical/similar values, or that want
    /// to look at values for multiple attributes, should rather get all values
    /// using [`MemoryAttribute::value()`] and manually select the target they
    /// consider the best.
    ///
    /// # Errors
    ///
    /// - [`ForeignInitiator`] if the `initiator` parameter was set to a
    ///   [`TopologyObject`] that does not belong to this topology
    /// - [`NeedInitiator`] if no `initiator` was provided but this memory
    ///   attribute needs one
    /// - [`UnwantedInitiator`] if an `initiator` was provided but this memory
    ///   attribute doesn't need one
    ///
    /// [`ForeignInitiator`]: InitiatorInputError::ForeignInitiator
    /// [`NeedInitiator`]: InitiatorInputError::NeedInitiator
    /// [`UnwantedInitiator`]: InitiatorInputError::UnwantedInitiator
    #[doc(alias = "hwloc_memattr_get_best_target")]
    pub fn best_target(
        &self,
        initiator: Option<MemoryAttributeLocation<'_>>,
    ) -> Result<Option<(&'topology TopologyObject, u64)>, InitiatorInputError> {
        // Validate the query
        // SAFETY: Will only be used before returning from this function,
        let initiator = unsafe { self.checked_initiator(initiator, false)? };

        // Run the query
        let mut best_target = ptr::null();
        // SAFETY: - hwloc_memattr_get_best_target is a "best X" query
        //         - Parameters are forwarded in the right order
        //         - initiator has been checked to come from this topology and
        //           to be NULL if and only if the attribute has no initiator
        //         - best_target is an out parameter, its initial value isn't read
        let opt = unsafe {
            self.get_best(
                "hwloc_memattr_get_best_target",
                |topology, attribute, flags, value| {
                    hwlocality_sys::hwloc_memattr_get_best_target(
                        topology,
                        attribute,
                        &initiator,
                        flags,
                        &mut best_target,
                        value,
                    )
                },
            )
        };

        // Convert target node into a safe high-level form
        // SAFETY: Target originates from a query against this topology
        Ok(opt.map(|value| (unsafe { self.encapsulate_target_node(best_target) }, value)))
    }

    /// Best initiator and associated attribute value, if any, for a given target node
    ///
    /// If multiple initiators have the same attribute values, only one is
    /// returned (and there is no way to clarify how that one is chosen).
    /// Applications that want to detect initiators with identical/similar
    /// values, or that want to look at values for multiple attributes, should
    /// rather get all values using [`MemoryAttribute::value()`] and manually
    /// select the initiator they consider the best.
    ///
    /// # Errors
    ///
    /// - [`NoInitiators`] if this memory attribute doesn't have initiators
    /// - [`ForeignTarget`] if `target` does not belong to this topology
    ///
    /// [`ForeignTarget`]: InitiatorQueryError::ForeignTarget
    /// [`NoInitiators`]: InitiatorQueryError::NoInitiators
    #[doc(alias = "hwloc_memattr_get_best_initiator")]
    pub fn best_initiator(
        &self,
        target: &TopologyObject,
    ) -> Result<Option<(MemoryAttributeLocation<'topology>, u64)>, InitiatorQueryError> {
        // Validate the query
        self.check_initiator_query_target(target)?;

        // Run the query
        // SAFETY: This is an out parameter, initial value won't be read
        let mut best_initiator = NULL_LOCATION;
        // SAFETY: - hwloc_memattr_get_best_initiator is a "best X" query
        //         - Parameters are forwarded in the right order
        //         - target node has been checked to come from this topology
        //         - best_initiator is an out parameter, its initial value isn't read
        let opt = unsafe {
            self.get_best(
                "hwloc_memattr_get_best_initiator",
                |topology, attribute, flags, value| {
                    hwlocality_sys::hwloc_memattr_get_best_initiator(
                        topology,
                        attribute,
                        target.as_inner(),
                        flags,
                        &mut best_initiator,
                        value,
                    )
                },
            )
        };

        // Convert initiator into a safe high-level form
        // SAFETY: Initiator originates from a query against this topology
        opt.map(|value| Ok((unsafe { self.encapsulate_initiator(best_initiator) }, value)))
            .transpose()
    }

    /// Target NUMA nodes that have some values for a given attribute, along
    /// with the associated values if possible.
    ///
    /// An `initiator` may only be specified if this attribute has the flag
    /// [`MemoryAttributeFlags::NEED_INITIATOR`]. In that case, specifying an
    /// initiator is optional and does two things:
    ///
    /// 1. It acts as a filter to only report targets that have a value for this
    ///    initiator.
    /// 2. It reports the memory attribute values for each listed target and
    ///    this initiator. Otherwise no values are reported, i.e. the output
    ///    `Option<Vec<u64>>` is `None`.
    ///
    /// The initiator should be a [`CpuSet`] when refering to accesses performed
    /// by CPU cores. [`MemoryAttributeLocation::Object`] is currently unused
    /// internally by hwloc, but user-defined memory attributes may for instance
    /// use it to provide custom information about host memory accesses
    /// performed by GPUs.
    ///
    /// In the case of memory attributes that have no initiators, `initiator`
    /// should be `None`, and memory attribute values will always be reported.
    ///
    /// This function is meant for tools and debugging (listing internal
    /// information) rather than for application queries. Applications should
    /// rather select useful NUMA nodes with [`Topology::local_numa_nodes()`]
    /// and then look at their attribute values.
    ///
    /// # Errors
    ///
    /// - [`ForeignInitiator`] if the `initiator` parameter was set to a
    ///   [`TopologyObject`] that does not belong to this topology
    /// - [`UnwantedInitiator`] if an `initiator` was provided but this memory
    ///   attribute doesn't need one
    ///
    /// [`ForeignInitiator`]: InitiatorInputError::ForeignInitiator
    /// [`UnwantedInitiator`]: InitiatorInputError::UnwantedInitiator
    #[allow(clippy::type_complexity)]
    #[doc(alias = "hwloc_memattr_get_targets")]
    pub fn targets(
        &self,
        initiator: Option<MemoryAttributeLocation<'_>>,
    ) -> Result<(Vec<&'topology TopologyObject>, Option<Vec<u64>>), HybridError<InitiatorInputError>>
    {
        // Validate the query + translate initiator to hwloc format
        let get_values =
            initiator.is_some() || !self.flags().contains(MemoryAttributeFlags::NEED_INITIATOR);
        // SAFETY: - Will only be used before returning from this function,
        //         - get_targets is documented to accept a NULL initiator
        let initiator = unsafe { self.checked_initiator(initiator, true)? };

        // Run the query
        // SAFETY: - hwloc_memattr_get_targets is indeed an array query
        //         - Parameters are forwarded in the right order
        //         - initiator has been checked to come from this topology and
        //           is allowed by the API contract to be NULL
        let (targets, values) = unsafe {
            self.array_query(
                "hwloc_memattr_get_targets",
                ptr::null(),
                get_values,
                |topology, attribute, flags, nr, targets, values| {
                    hwlocality_sys::hwloc_memattr_get_targets(
                        topology, attribute, &initiator, flags, nr, targets, values,
                    )
                },
            )
            .map_err(HybridError::Hwloc)?
        };

        // Convert target list into a safe high-level form
        let targets = targets
            .into_iter()
            // SAFETY: Targets originate from a query against this topology
            .map(|node_ptr| unsafe { self.encapsulate_target_node(node_ptr) })
            .collect();
        let values = (!values.is_empty()).then_some(values);
        Ok((targets, values))
    }

    /// Initiators that have values for a given attribute for a specific target
    /// NUMA node, along with the associated values
    ///
    /// This function is meant for tools and debugging (listing internal
    /// information) rather than for application queries. Applications should
    /// rather select useful NUMA nodes with [`Topology::local_numa_nodes()`]
    /// and then look at their attribute values.
    ///
    /// # Errors
    ///
    /// - [`NoInitiators`] if this memory attribute doesn't have initiators
    /// - [`ForeignTarget`] if `target` does not belong to this topology
    ///
    /// [`ForeignTarget`]: InitiatorQueryError::ForeignTarget
    /// [`NoInitiators`]: InitiatorQueryError::NoInitiators
    #[doc(alias = "hwloc_memattr_get_initiators")]
    pub fn initiators(
        &self,
        target_node: &TopologyObject,
    ) -> Result<(Vec<MemoryAttributeLocation<'topology>>, Vec<u64>), HybridError<InitiatorQueryError>>
    {
        // Validate the query
        self.check_initiator_query_target(target_node)?;

        // Run the query
        // SAFETY: - hwloc_memattr_get_initiators is indeed an array query
        //         - Parameters are forwarded in the right order
        //         - target_node has been checked to come from this topology
        let (initiators, values) = unsafe {
            self.array_query(
                "hwloc_memattr_get_initiators",
                NULL_LOCATION,
                true,
                |topology, attribute, flags, nr, initiators, values| {
                    hwlocality_sys::hwloc_memattr_get_initiators(
                        topology,
                        attribute,
                        target_node.as_inner(),
                        flags,
                        nr,
                        initiators,
                        values,
                    )
                },
            )
            .map_err(HybridError::Hwloc)?
        };

        // Convert initiators into a safe high-level form
        let initiators = initiators
            .into_iter()
            // SAFETY: Initiators originate from a query against this topology
            .map(|initiator| unsafe { self.encapsulate_initiator(initiator) })
            .collect();
        Ok((initiators, values))
    }

    /// Perform a `get_targets` style memory attribute query
    ///
    /// `get_values` indicates whether the caller is interested in associated
    /// memory attribute values. If so, the output `Vec<u64>` will be filled
    /// with as many values as the number of initiators/targets, if not it will
    /// be an empty `Vec`.
    ///
    /// # Safety
    ///
    /// `query` must be one of the `hwloc_memattr_<plural>` queries, with the
    /// parameter list simplified to the elements that are common to all of
    /// these queries:
    ///
    /// - Topology
    /// - Memory attribute id
    /// - Flags
    /// - In/out number of memory attribute values
    /// - Output endpoint buffer with capacity given above
    /// - Output value buffer with capacity given above
    unsafe fn array_query<Endpoint: Copy>(
        &self,
        api: &'static str,
        placeholder: Endpoint,
        get_values: bool,
        mut query: impl FnMut(
            hwloc_const_topology_t,
            hwloc_memattr_id_t,
            c_ulong,
            *mut c_uint,
            *mut Endpoint,
            *mut u64,
        ) -> c_int,
    ) -> Result<(Vec<Endpoint>, Vec<u64>), RawHwlocError> {
        // Prepare to call hwloc
        let mut call_ffi = |nr_mut, endpoint_out, values_out| {
            errors::call_hwloc_int_normal(api, || {
                // SAFETY: - Topology is trusted to contain a valid ptr (type invariant)
                //         - hwloc ops are trusted not to modify *const parameters
                //         - id is trusted to be valid (type invariant)
                //         - flags must be 0 for all of these queries
                //         - Correctness of nr_mut, enpoint_out and values_out
                //           is call site dependent, see below
                query(
                    self.topology.as_ptr(),
                    self.id,
                    0,
                    nr_mut,
                    endpoint_out,
                    values_out,
                )
            })
        };

        // Determine the right size for arrays of endpoints and values
        // SAFETY: 1 elements + throw-away buffers is the correct way to request
        //         the buffer size to be allocated from hwloc
        let mut nr = 1;
        let res = call_ffi(&mut nr, [placeholder].as_mut_ptr(), [u64::MAX].as_mut_ptr());
        let len = int::expect_usize(nr);
        let mut endpoints = vec![placeholder; len];

        // At the time of writing, all error cases of hwloc_memattr_get_targets
        // are either avoided through API design or checked before attempting to
        // call array_query(), so errors can only emerged through hwloc changes.
        // But this is not true for hwloc_memattr_get_initiators. In that case,
        // EINVAL is returned if the specified target has no memory attribute
        // attached to it. This is only subtly different from having a memory
        // attribute defined for this target, but no value/initiator attached,
        // which yields no errors (just an empty initiator/value list), so we
        // choose to handle it identically by returning an empty list.
        if api == "hwloc_memattr_get_initiators" {
            match res {
                Ok(_positive) => {}
                Err(RawHwlocError {
                    errno: Some(Errno(EINVAL)),
                    ..
                }) => return Ok((Vec::new(), Vec::new())),
                #[cfg(windows)]
                Err(RawHwlocError { errno: None, .. }) => {
                    // As explained in the RawHwlocError documentation, errno values
                    // may not correctly propagate from hwloc to hwlocality on
                    // Windows. But only EINVAL is expected here so we're fine.
                    return Ok((Vec::new(), Vec::new()));
                }
                Err(other_err) => return Err(other_err),
            }
        } else {
            res?;
        }

        // Get the values if requested
        let old_nr = nr;
        let values = if get_values {
            let mut values = vec![u64::MAX; len];
            // SAFETY: - endpoints and values are indeed arrays of nr = len elements
            //         - Input array contents don't matter as this is an out-parameter
            call_ffi(&mut nr, endpoints.as_mut_ptr(), values.as_mut_ptr())?;
            values
        } else {
            // SAFETY: - endpoints is indeed an array of nr = len elements,
            //           values can be null to indicate lack of interest
            //         - Input array contents don't matter as this is an out-parameter
            call_ffi(&mut nr, endpoints.as_mut_ptr(), ptr::null_mut())?;
            Vec::new()
        };
        assert_eq!(old_nr, nr, "Inconsistent node count from hwloc");
        Ok((endpoints, values))
    }

    /// Perform a `get_best_initiator`-style memory attribute query, assuming
    /// the query arguments have been checked for correctness
    ///
    /// # Safety
    ///
    /// `query` must be one of the `hwloc_memattr_get_best_` queries, with the
    /// parameter list simplified to the elements that are common to all of
    /// these queries:
    ///
    /// - Topology
    /// - Memory attribute id
    /// - Flags
    /// - Best value output
    unsafe fn get_best(
        &self,
        api: &'static str,
        query: impl FnOnce(hwloc_const_topology_t, hwloc_memattr_id_t, c_ulong, *mut u64) -> c_int,
    ) -> Option<u64> {
        /// Polymorphized subset of this function (avoids generics code bloat)
        fn process_result(
            api: &'static str,
            final_value: u64,
            result: Result<c_uint, RawHwlocError>,
        ) -> Option<u64> {
            match result {
                Ok(_positive) => Some(final_value),
                Err(RawHwlocError {
                    errno: Some(Errno(ENOENT)),
                    ..
                }) => None,
                // EINVAL can mean several different things here:
                // - Queried an initiator on a memory attribute that doesn't
                //   have them. This error is handled before calling this, so we
                //   should never observe an associated EINVAL.
                // - Flags are nonzero. We prevent this by not letting the user
                //   set flags.
                // - Memory attribute ID is invalid. We prevent this by not
                //   letting the user create a MemoryAttribute with an invalid
                //   ID.
                // - Requested best initiator on a target that is not associated
                //   with this memory attribute. This error is only subtly
                //   different fom ENOENT which means that the target is
                //   registered but no initiator is registered. This is also
                //   inconsistent with best_target queries which return ENOENT
                //   in the case of a nonexistent initiator. Since Windows may
                //   not know the difference anyway (see below), we do not
                //   expose this distinction in the API and handle every case
                //   where there is no "best X for this Y" by returning None.
                Err(RawHwlocError {
                    errno: Some(Errno(EINVAL)),
                    ..
                }) if api == "hwloc_memattr_get_best_initiator" => None,
                #[cfg(windows)]
                Err(RawHwlocError { errno: None, .. }) => {
                    // As explained in the RawHwlocError documentation, errno values
                    // may not correctly propagate from hwloc to hwlocality on
                    // Windows. But only EINVAL and ENOENT are expected here,
                    // and we handle them identically above, so we're fine.
                    None
                }
                Err(raw_err) => unreachable!("Unexpected hwloc error: {raw_err}"),
            }
        }

        // Call hwloc and process its results
        let mut value = u64::MAX;
        let result = errors::call_hwloc_int_normal(api, || {
            // SAFETY: - Topology is trusted to contain a valid ptr (type invariant)
            //         - hwloc ops are trusted not to modify *const parameters
            //         - id is trusted to be valid (type invariant)
            //         - flags must be 0 for all of these queries
            //         - value is an out-parameter, input value doesn't matter
            query(self.topology.as_ptr(), self.id, 0, &mut value)
        });
        process_result(api, value, result)
    }

    /// Encapsulate a `*const TopologyObject` to a target NUMA node from hwloc
    ///
    /// # Safety
    ///
    /// `node_ptr` must originate from a query against this attribute
    unsafe fn encapsulate_target_node(
        &self,
        node_ptr: *const hwloc_obj,
    ) -> &'topology TopologyObject {
        assert!(!node_ptr.is_null(), "Got null target pointer from hwloc");
        // SAFETY: Lifetime per input precondition, query output assumed valid
        unsafe { (&*node_ptr).as_newtype() }
    }

    /// Encapsulate an initiator location from hwloc
    ///
    /// # Safety
    ///
    /// `initiator` must originate a query against this attribute
    unsafe fn encapsulate_initiator(
        &self,
        initiator: hwloc_location,
    ) -> MemoryAttributeLocation<'topology> {
        // SAFETY: Lifetime per input precondition, query output assumed valid
        unsafe { MemoryAttributeLocation::from_raw(self.topology, initiator) }
            .expect("Failed to decode location from hwloc")
    }

    /// Check the initiator argument to some query, then translate it into the
    /// lower-level format that hwloc expects
    ///
    /// If `is_optional` is true, it is okay not to provide an initiator even
    /// if the memory attribute has flag [`MemoryAttributeFlags::NEED_INITIATOR`].
    ///
    /// # Safety
    ///
    /// - Do not use the output after the `'initiator` lifetime has expired.
    /// - `is_optional` should only be set to `true` for recipients that are
    ///   documented to accept NULL initiators.
    #[allow(clippy::needless_lifetimes)]
    unsafe fn checked_initiator<'initiator>(
        &self,
        initiator: Option<MemoryAttributeLocation<'initiator>>,
        is_optional: bool,
    ) -> Result<hwloc_location, InitiatorInputError> {
        // Collect flags
        let flags = self.flags();

        // Handle missing initiator case
        if flags.contains(MemoryAttributeFlags::NEED_INITIATOR)
            && initiator.is_none()
            && !is_optional
        {
            return Err(InitiatorInputError::NeedInitiator(self.error_name()));
        }

        // Handle unexpected initiator case
        if !flags.contains(MemoryAttributeFlags::NEED_INITIATOR) && initiator.is_some() {
            return Err(InitiatorInputError::UnwantedInitiator(self.error_name()));
        }

        // Handle expected absence of initiator
        let Some(initiator) = initiator else {
            // SAFETY: Per input precondition of is_optional + check above that
            //         initiator can only be none if initiator is optional
            return Ok(NULL_LOCATION);
        };

        // Make sure initiator does belong to this topology
        // SAFETY: Per function precondition on output usage
        unsafe {
            initiator
                .to_checked_raw(self.topology)
                .map_err(InitiatorInputError::ForeignInitiator)
        }
    }

    /// Check the target argument to some initiator query
    fn check_initiator_query_target(
        &self,
        target: &TopologyObject,
    ) -> Result<(), InitiatorQueryError> {
        if !self.flags().contains(MemoryAttributeFlags::NEED_INITIATOR) {
            return Err(InitiatorQueryError::NoInitiators(self.error_name()));
        }
        if !self.topology.contains(target) {
            return Err(target.into());
        }
        Ok(())
    }

    /// Translate this attribute's name to a form suitable for error reporting
    fn error_name(&self) -> Box<str> {
        self.name().to_string_lossy().into()
    }
}
//
impl Debug for MemoryAttribute<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = format!("{}(#{})", self.name().to_string_lossy(), self.id);
        f.pad(&name)
    }
}

/// An invalid initiator was passed to a memory attribute function
#[derive(Clone, Debug, Eq, Error, Hash, PartialEq)]
pub enum InitiatorInputError {
    /// Provided initiator is a [`TopologyObject`] that does not belong to this
    /// topology
    #[error("memory attribute initiator {0}")]
    ForeignInitiator(#[from] ForeignObjectError),

    /// An initiator had to be provided, but was not provided
    #[error("memory attribute {0} needs an initiator but none was provided")]
    NeedInitiator(Box<str>),

    /// No initiator should have been provided, but one was provided
    #[error("memory attribute {0} has no initiator but an initiator was provided")]
    UnwantedInitiator(Box<str>),
}
//
impl<'topology> From<&'topology TopologyObject> for InitiatorInputError {
    fn from(object: &'topology TopologyObject) -> Self {
        Self::ForeignInitiator(object.into())
    }
}

/// A query for memory attribute initiators is invalid
#[derive(Clone, Debug, Eq, Error, Hash, PartialEq)]
pub enum InitiatorQueryError {
    /// The specified `target` object does not belong to this topology
    #[error("memory attribute target {0}")]
    ForeignTarget(#[from] ForeignObjectError),

    /// This memory attribute doesn't have initiators
    #[error("memory attribute {0} has no initiator but its initiator was queried")]
    NoInitiators(Box<str>),
}
//
impl<'topology> From<&'topology TopologyObject> for InitiatorQueryError {
    fn from(object: &'topology TopologyObject) -> Self {
        Self::ForeignTarget(object.into())
    }
}

/// A memory attribute value query is invalid
#[derive(Clone, Debug, Eq, Error, Hash, PartialEq)]
pub enum ValueQueryError {
    /// Specified `initiator` is bad
    #[error(transparent)]
    BadInitiator(#[from] InitiatorInputError),

    /// Specified target object does not belong to this topology
    #[error("memory attribute target {0}")]
    ForeignTarget(ForeignObjectError),
}

/// Where to measure attributes from
#[doc(alias = "hwloc_location")]
#[doc(alias = "hwloc_location::location")]
#[doc(alias = "hwloc_location::type")]
#[doc(alias = "hwloc_location_u")]
#[doc(alias = "hwloc_location::hwloc_location_u")]
#[doc(alias = "hwloc_location_type_e")]
#[derive(Clone, Copy, Debug, Display)]
pub enum MemoryAttributeLocation<'target> {
    /// Directly provide CPU set to find NUMA nodes with corresponding locality
    ///
    /// This is the only initiator type supported by most memory attribute
    /// queries on hwloc-defined memory attributes, though `Object` remains an
    /// option for user-defined memory attributes.
    #[doc(alias = "HWLOC_LOCATION_TYPE_CPUSET")]
    CpuSet(BitmapRef<'target, CpuSet>),

    /// Use a topology object as an initiator
    ///
    /// Most memory attribute queries on hwloc-defined memory attributes do not
    /// support this initiator type, or translate it to a cpuset (going up the
    /// ancestor chain if necessary). But user-defined memory attributes may for
    /// instance use it to provide custom information about host memory accesses
    /// performed by GPUs.
    ///
    /// Only objects belonging to the topology to which memory attributes are
    /// attached should be used here.
    #[doc(alias = "HWLOC_LOCATION_TYPE_OBJECT")]
    Object(&'target TopologyObject),
}
//
impl<'target> MemoryAttributeLocation<'target> {
    /// Convert to the C representation for the purpose of running an hwloc
    /// query against `topology`.
    ///
    /// # Errors
    ///
    /// [`ForeignObjectError`] if this [`MemoryAttributeLocation`] is constructed
    /// from an `&TopologyObject` that does not belong to the target [`Topology`].
    ///
    /// # Safety
    ///
    /// Do not use the output after the source lifetime has expired
    pub(crate) unsafe fn to_checked_raw(
        self,
        topology: &Topology,
    ) -> Result<hwloc_location, ForeignObjectError> {
        match self {
            Self::CpuSet(cpuset) => Ok(hwloc_location {
                ty: HWLOC_LOCATION_TYPE_CPUSET,
                location: hwloc_location_u {
                    cpuset: cpuset.as_ptr(),
                },
            }),
            Self::Object(object) => {
                if topology.contains(object) {
                    Ok(hwloc_location {
                        ty: HWLOC_LOCATION_TYPE_OBJECT,
                        location: hwloc_location_u {
                            object: object.as_inner(),
                        },
                    })
                } else {
                    Err(object.into())
                }
            }
        }
    }

    /// Convert from the C representation
    ///
    /// # Safety
    ///
    /// This function should only be used to encapsulate [`hwloc_location`] structs
    /// from hwloc topology queries, and the `_topology` parameter should match
    /// the [`Topology`] from which the location was extracted.
    unsafe fn from_raw(
        _topology: &'target Topology,
        raw: hwloc_location,
    ) -> Result<Self, LocationTypeError> {
        // SAFETY: - Location type information from hwloc is assumed to be
        //           correct and tells us which union variant we should read.
        //         - Pointer is assumed to point to a valid CpuSet or
        //           TopologyObject that is owned by _topology, and thus has a
        //           lifetime of 'target or greater.
        unsafe {
            match raw.ty {
                HWLOC_LOCATION_TYPE_CPUSET => {
                    let ptr = NonNull::new(raw.location.cpuset.cast_mut())
                        .expect("Unexpected null CpuSet from hwloc");
                    let cpuset = CpuSet::borrow_from_nonnull(ptr);
                    Ok(MemoryAttributeLocation::CpuSet(cpuset))
                }
                HWLOC_LOCATION_TYPE_OBJECT => {
                    let ptr = NonNull::new(raw.location.object.cast_mut())
                        .expect("Unexpected null TopologyObject from hwloc");
                    Ok(MemoryAttributeLocation::Object(ptr.as_ref().as_newtype()))
                }
                unknown => Err(LocationTypeError(unknown)),
            }
        }
    }
}
//
impl<'target> From<BitmapRef<'target, CpuSet>> for MemoryAttributeLocation<'target> {
    fn from(cpuset: BitmapRef<'target, CpuSet>) -> Self {
        Self::CpuSet(cpuset)
    }
}
//
impl<'target> From<&'target CpuSet> for MemoryAttributeLocation<'target> {
    fn from(cpuset: &'target CpuSet) -> Self {
        BitmapRef::from(cpuset).into()
    }
}
//
impl<'target> From<&'target TopologyObject> for MemoryAttributeLocation<'target> {
    fn from(object: &'target TopologyObject) -> Self {
        Self::Object(object)
    }
}

/// Error returned when an unknown location type is observed
#[derive(Copy, Clone, Debug, Eq, Error, From, Hash, PartialEq)]
#[error("hwloc provided a memory attribute location of unknown type {0}")]
struct LocationTypeError(c_int);

/// Invalid `hwloc_location`, which hwloc is assumed not to observe
///
/// # Safety
///
/// Do not expose this location value to an hwloc function that actually
/// reads it, unless it is explicitly documented to accept NULL locations.
const NULL_LOCATION: hwloc_location = hwloc_location {
    ty: HWLOC_LOCATION_TYPE_CPUSET,
    location: hwloc_location_u {
        cpuset: ptr::null(),
    },
};

bitflags! {
    /// Flags for selecting more target NUMA nodes
    ///
    /// By default only NUMA nodes whose locality is exactly the given location
    /// are selected.
    #[derive(Copy, Clone, Debug, Default, Eq, Hash, PartialEq)]
    #[doc(alias = "hwloc_local_numanode_flag_e")]
    pub struct LocalNUMANodeFlags: hwloc_local_numanode_flag_e {
        /// Select NUMA nodes whose locality is larger than the given cpuset
        ///
        /// For instance, if a single PU (or its cpuset) is given in `initiator`,
        /// select all nodes close to the package that contains this PU.
        #[doc(alias = "HWLOC_LOCAL_NUMANODE_FLAG_LARGER_LOCALITY")]
        const LARGER_LOCALITY = HWLOC_LOCAL_NUMANODE_FLAG_LARGER_LOCALITY;

        /// Select NUMA nodes whose locality is smaller than the given cpuset
        ///
        /// For instance, if a package (or its cpuset) is given in `initiator`,
        /// also select nodes that are attached to only a half of that package.
        #[doc(alias = "HWLOC_LOCAL_NUMANODE_FLAG_SMALLER_LOCALITY")]
        const SMALLER_LOCALITY = HWLOC_LOCAL_NUMANODE_FLAG_SMALLER_LOCALITY;

        /// Select all NUMA nodes in the topology
        ///
        /// The initiator is ignored.
        ///
        /// This flag is automatically set when users specify
        /// [`NUMALocation::All`] as the target NUMA node set.
        #[doc(hidden)]
        const ALL = HWLOC_LOCAL_NUMANODE_FLAG_ALL;
    }
}
//
crate::impl_arbitrary_for_bitflags!(LocalNUMANodeFlags, hwloc_local_numanode_flag_e);

/// Scope of a [`Topology::local_numa_nodes()`] query
#[derive(Copy, Clone, Debug)]
pub enum NUMALocation<'target> {
    /// Nodes local to some object
    Local {
        /// Initiator which NUMA nodes should be local to
        ///
        /// By default, the search only returns NUMA nodes whose locality is
        /// exactly the given `location`. More nodes can be selected using
        /// `flags`.
        location: MemoryAttributeLocation<'target>,

        /// Flags for enlarging the NUMA node search
        flags: LocalNUMANodeFlags,
    },

    /// All NUMA nodes in the topology
    #[doc(alias = "HWLOC_LOCAL_NUMANODE_FLAG_ALL")]
    All,
}
//
impl NUMALocation<'_> {
    /// Convert to the inputs expected by a `hwloc_get_local_numanode_objs`
    /// query against `topology`
    ///
    /// # Errors
    ///
    /// [`ForeignObjectError`] if the input location is a [`TopologyObject`] that
    /// does not belong to the target [`Topology`]
    ///
    /// # Safety
    ///
    /// Do not use the output raw location after the source lifetime has expired
    pub(crate) unsafe fn to_checked_raw(
        self,
        topology: &Topology,
    ) -> Result<(hwloc_location, LocalNUMANodeFlags), ForeignObjectError> {
        match self {
            Self::Local {
                location,
                mut flags,
            } => {
                flags.remove(LocalNUMANodeFlags::ALL);
                // SAFETY: Per function precondition
                Ok((unsafe { location.to_checked_raw(topology)? }, flags))
            }
            // SAFETY: In presence of the ALL flag, the initiator is ignored,
            //         so a null location is fine.
            Self::All => Ok((NULL_LOCATION, LocalNUMANodeFlags::ALL)),
        }
    }
}
//
impl<'target, T: Into<MemoryAttributeLocation<'target>>> From<T> for NUMALocation<'target> {
    fn from(location: T) -> Self {
        Self::Local {
            location: location.into(),
            flags: LocalNUMANodeFlags::empty(),
        }
    }
}

bitflags! {
    /// Memory attribute flags
    ///
    /// Exactly one of the `HIGHER_IS_BEST` and `LOWER_IS_BEST` flags must be set.
    #[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
    #[doc(alias = "hwloc_memattr_flag_e")]
    pub struct MemoryAttributeFlags: hwloc_memattr_flag_e {
        /// The best nodes for this memory attribute are those with the higher
        /// values
        ///
        /// This flag is mutually exclusive with [`LOWER_IS_BEST`].
        ///
        /// See for instance [`MemoryAttribute::bandwidth()`].
        ///
        /// [`LOWER_IS_BEST`]: Self::LOWER_IS_BEST
        #[doc(alias = "HWLOC_MEMATTR_FLAG_HIGHER_FIRST")]
        const HIGHER_IS_BEST = HWLOC_MEMATTR_FLAG_HIGHER_FIRST;

        /// The best nodes for this memory attribute are those with the lower
        /// values
        ///
        /// This flag is mutually exclusive with [`HIGHER_IS_BEST`].
        ///
        /// See for instance [`MemoryAttribute::latency()`].
        ///
        /// [`HIGHER_IS_BEST`]: Self::HIGHER_IS_BEST
        #[doc(alias = "HWLOC_MEMATTR_FLAG_LOWER_FIRST")]
        const LOWER_IS_BEST = HWLOC_MEMATTR_FLAG_LOWER_FIRST;

        /// The value returned for this memory attribute depends on the given
        /// initiator
        ///
        /// See for instance [`bandwidth()`] and [`latency()`], but not
        /// [`capacity()`].
        ///
        /// [`bandwidth()`]: MemoryAttribute::bandwidth()
        /// [`latency()`]: MemoryAttribute::latency()
        /// [`capacity()`]: MemoryAttribute::capacity()
        #[doc(alias = "HWLOC_MEMATTR_FLAG_NEED_INITIATOR")]
        const NEED_INITIATOR = HWLOC_MEMATTR_FLAG_NEED_INITIATOR;
    }
}
//
impl MemoryAttributeFlags {
    /// Truth that these flags are in a valid state
    pub(crate) fn is_valid(self) -> bool {
        self.contains(Self::HIGHER_IS_BEST) ^ self.contains(Self::LOWER_IS_BEST)
    }
}
//
crate::impl_arbitrary_for_bitflags!(MemoryAttributeFlags, hwloc_memattr_flag_e);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        object::types::ObjectType,
        strategies::{any_object, any_string, set_with_reference, topology_related_set},
    };
    use proptest::prelude::*;
    #[allow(unused)]
    use similar_asserts::assert_eq;
    use std::{cmp::Ordering, collections::HashMap, ffi::CString, sync::OnceLock};

    /// Mechanism to track the best value of a memory attribute and the
    /// endpoints for which the memory attribute has this value.
    struct BestValueEndpoints<Endpoint> {
        higher_is_best: bool,
        best_value: u64,
        best_endpoints: Vec<Endpoint>,
    }
    //
    impl<Endpoint: Copy> BestValueEndpoints<Endpoint> {
        fn new(higher_is_best: bool) -> Self {
            Self {
                higher_is_best,
                best_value: if higher_is_best { u64::MIN } else { u64::MAX },
                best_endpoints: Vec::new(),
            }
        }

        fn push(&mut self, endpoint: Endpoint, value: u64) {
            match (self.higher_is_best, value.cmp(&self.best_value)) {
                (true, Ordering::Greater) | (false, Ordering::Less) => {
                    self.best_value = value;
                    self.best_endpoints.clear();
                    self.best_endpoints.push(endpoint);
                }
                (_, Ordering::Equal) => {
                    self.best_endpoints.push(endpoint);
                }
                _ => {}
            }
        }

        fn collect_value_and_endpoints(self) -> Option<(Vec<Endpoint>, u64)> {
            (!self.best_endpoints.is_empty()).then_some((self.best_endpoints, self.best_value))
        }
    }

    /// Unique initiator identifier
    #[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
    enum InitiatorKey<'a> {
        CpuSet(BitmapRef<'a, CpuSet>),
        Object(TopologyObjectID),
    }
    //
    impl<'a> From<MemoryAttributeLocation<'a>> for InitiatorKey<'a> {
        fn from(x: MemoryAttributeLocation<'a>) -> Self {
            match x {
                MemoryAttributeLocation::CpuSet(set) => Self::CpuSet(set),
                MemoryAttributeLocation::Object(obj) => Self::Object(obj.global_persistent_index()),
            }
        }
    }

    /// Test all memory attributes for all queries that should succeed
    #[test]
    fn successful_queries() {
        let topology = Topology::test_instance();
        for attr in MemoryAttribute::all(topology) {
            let name = attr.name().to_str().unwrap();
            let by_name = topology
                .memory_attribute_named(name)
                .expect("Memory attributes should not have a NUL in their name")
                .expect("MemoryAttribute::all() should not yield nonexistent attributes");
            assert!(ptr::eq(by_name.topology, attr.topology));
            assert_eq!(by_name.id, attr.id);

            let flags = attr.flags();
            assert_eq!(
                flags
                    .intersection(
                        MemoryAttributeFlags::HIGHER_IS_BEST | MemoryAttributeFlags::LOWER_IS_BEST
                    )
                    .iter()
                    .count(),
                1
            );

            if flags.contains(MemoryAttributeFlags::NEED_INITIATOR) {
                test_targets_with_initiators(attr);
            } else {
                test_targets_wo_initiators(attr);
            }
        }
    }

    /// Target and initiator testing for memory attributes that need an
    /// initiator
    fn test_targets_with_initiators(attr_with_initiator: MemoryAttribute<'_>) {
        // Check basic attribute properties and query targets
        let attr = attr_with_initiator;
        let higher_is_best = attr.flags().contains(MemoryAttributeFlags::HIGHER_IS_BEST);
        let (target_nodes, no_values) = attr.targets(None).unwrap();
        assert!(no_values.is_none());

        // For each target...
        let mut key_to_initiator = HashMap::new();
        let mut initiator_key_to_target_ids_and_values = HashMap::<_, HashSet<_>>::new();
        for target in target_nodes {
            // Check initiators, collect best initiator and targets for
            // each initiator
            let (initiators, values) = attr.initiators(target).unwrap();
            let mut expected_best_initiators = BestValueEndpoints::new(higher_is_best);
            for (initiator, value) in initiators.into_iter().zip(values) {
                assert_eq!(attr.value(Some(initiator), target), Ok(Some(value)));
                let initiator_key = InitiatorKey::from(initiator);
                key_to_initiator.insert(initiator_key, initiator);
                let inserted = initiator_key_to_target_ids_and_values
                    .entry(initiator_key)
                    .or_default()
                    .insert((target.global_persistent_index(), value));
                assert!(inserted);
                expected_best_initiators.push(initiator, value);
            }

            // Check best initiator
            let expected_best_initiators = expected_best_initiators.collect_value_and_endpoints();
            let best_initiator = attr.best_initiator(target).unwrap();
            assert_eq!(expected_best_initiators.is_none(), best_initiator.is_none());
            if let (
                Some((expected_best_initiators, expected_best_value)),
                Some((best_initiator, best_value)),
            ) = (expected_best_initiators, best_initiator)
            {
                assert_eq!(best_value, expected_best_value);
                assert!(expected_best_initiators.into_iter().any(|initiator| {
                    match (initiator, best_initiator) {
                        (
                            MemoryAttributeLocation::CpuSet(set1),
                            MemoryAttributeLocation::CpuSet(set2),
                        ) => set1 == set2,
                        (
                            MemoryAttributeLocation::Object(obj1),
                            MemoryAttributeLocation::Object(obj2),
                        ) => obj1.global_persistent_index() == obj2.global_persistent_index(),
                        _ => false,
                    }
                }));
            }
        }

        // For each initiator...
        for (initiator_key, target_ids_and_values) in initiator_key_to_target_ids_and_values {
            // Check targets, find best targets
            let initiator = key_to_initiator[&initiator_key];
            let (targets, values) = attr.targets(Some(initiator)).unwrap();
            let values = values.unwrap();
            assert_eq!(targets.len(), values.len());
            assert_eq!(targets.len(), target_ids_and_values.len());
            let mut expected_best_targets = BestValueEndpoints::new(higher_is_best);
            for (target, value) in targets.iter().zip(values) {
                assert!(target_ids_and_values.contains(&(target.global_persistent_index(), value)));
                expected_best_targets.push(target.global_persistent_index(), value);
            }

            // Check best target
            let expected_best_targets = expected_best_targets.collect_value_and_endpoints();
            let best_target = attr.best_target(Some(initiator)).unwrap();
            assert_eq!(expected_best_targets.is_none(), best_target.is_none());
            if let (
                Some((expected_best_targets, expected_best_value)),
                Some((best_target, best_value)),
            ) = (expected_best_targets, best_target)
            {
                assert_eq!(best_value, expected_best_value);
                assert!(expected_best_targets
                    .into_iter()
                    .any(|target_id| { best_target.global_persistent_index() == target_id }));
            }
        }
    }

    /// Target and initiator testing for memory attributes that have no
    /// initiator
    fn test_targets_wo_initiators(attr_wo_initiator: MemoryAttribute<'_>) {
        // Check basic attribute properties and query targets
        let attr = attr_wo_initiator;
        let higher_is_best = attr.flags().contains(MemoryAttributeFlags::HIGHER_IS_BEST);
        let (target_nodes, values) = attr.targets(None).unwrap();
        let values_via_targets = values.unwrap();
        assert_eq!(target_nodes.len(), values_via_targets.len());

        // Check values, find best targets
        let mut expected_best_targets = BestValueEndpoints::new(higher_is_best);
        for (target, value) in target_nodes.into_iter().zip(values_via_targets) {
            assert_eq!(attr.value(None, target), Ok(Some(value)));
            expected_best_targets.push(target, value);
        }

        // Check best target
        let expected_best_targets = expected_best_targets.collect_value_and_endpoints();
        let best_target = attr.best_target(None).unwrap();
        assert_eq!(expected_best_targets.is_none(), best_target.is_none());
        if let (
            Some((expected_best_targets, expected_best_value)),
            Some((best_target, best_value)),
        ) = (expected_best_targets, best_target)
        {
            assert_eq!(best_value, expected_best_value);
            assert!(expected_best_targets
                .into_iter()
                .any(|target| target.global_persistent_index()
                    == best_target.global_persistent_index()));
        }
    }

    proptest! {
        #[test]
        fn memory_attribute_named(name in any_string()) {
            let topology = Topology::test_instance();
            let res = topology.memory_attribute_named(&name);

            let name_contains_nul = name.contains('\0');
            let Ok(maybe_attr) = res else {
                prop_assert!(name_contains_nul);
                return Ok(());
            };
            prop_assert!(!name_contains_nul);

            if maybe_attr.is_none() {
                let name = CString::new(name).unwrap();
                for attr in MemoryAttribute::all(topology) {
                    prop_assert_ne!(attr.name(), name.as_c_str());
                }
            }
        }
    }

    /// Pick a memory attribute
    fn memory_attribute() -> impl Strategy<Value = MemoryAttribute<'static>> {
        prop::sample::select(MemoryAttribute::all(Topology::test_instance()).collect::<Vec<_>>())
    }

    /// Pick a memory attribute and a target that has a chance of being correct
    /// for this attribute, but may be a random object possibly coming from
    /// another topology.
    fn memory_attribute_and_target(
    ) -> impl Strategy<Value = (MemoryAttribute<'static>, &'static TopologyObject)> {
        memory_attribute().prop_flat_map(|attr| {
            let targets = attr.targets(None).unwrap().0;
            let target = if targets.is_empty() {
                any_object().boxed()
            } else {
                prop_oneof![
                    2 => prop::sample::select(targets),
                    3 => any_object(),
                ]
                .boxed()
            };
            (Just(attr), target)
        })
    }

    proptest! {
        #[test]
        fn initiator_query_errors(
            (attr, target) in memory_attribute_and_target()
        ) {
            let no_initiators = !attr.flags().contains(MemoryAttributeFlags::NEED_INITIATOR);
            let foreign_target = !attr.topology.contains(target);
            let check_normalized_error = |res_wo_ok: Result<(), InitiatorQueryError>| {
                match res_wo_ok {
                    Ok(()) => prop_assert!(!(no_initiators || foreign_target)),
                    Err(InitiatorQueryError::ForeignTarget(_)) => prop_assert!(foreign_target),
                    Err(InitiatorQueryError::NoInitiators(_)) => prop_assert!(no_initiators),
                }
                Ok(())
            };
            check_normalized_error(attr.best_initiator(target).map(std::mem::drop))?;
            check_normalized_error(attr.initiators(target).map(std::mem::drop).map_err(|e| {
                let HybridError::Rust(e) = e else {
                    unreachable!("No known error cases that aren't handled on the Rust side, yet got {e:?}")
                };
                e
            }))?;
        }
    }

    /// Owned version of [`MemoryAttributeLocation`] for testing
    #[derive(Clone, Debug)]
    enum OwnedAttributeLocation {
        CpuSet(CpuSet),
        Object(&'static TopologyObject),
    }
    //
    fn borrow_owned_initiator(
        initiator: &Option<OwnedAttributeLocation>,
    ) -> Option<MemoryAttributeLocation<'_>> {
        match initiator {
            Some(OwnedAttributeLocation::CpuSet(set)) => Some(MemoryAttributeLocation::from(set)),
            Some(OwnedAttributeLocation::Object(obj)) => Some(MemoryAttributeLocation::from(*obj)),
            None => None,
        }
    }

    /// Pick a memory attribute initiator that has a low odd of being valid
    fn any_initiator() -> impl Strategy<Value = Option<OwnedAttributeLocation>> {
        prop_oneof![
            topology_related_set(Topology::cpuset)
                .prop_map(|set| Some(OwnedAttributeLocation::CpuSet(set))),
            any_object().prop_map(|obj| Some(OwnedAttributeLocation::Object(obj)))
        ]
    }

    /// Pick an (attribute, target, initiator) triple where (target, initiator)
    /// has a chance of being correct for this attribute, but may be a random
    /// object/cpuset possibly coming from another topology.
    fn memory_attribute_target_initiator() -> impl Strategy<
        Value = (
            MemoryAttribute<'static>,
            &'static TopologyObject,
            Option<OwnedAttributeLocation>,
        ),
    > {
        memory_attribute_and_target().prop_flat_map(move |(attr, target)| {
            let initiator = if attr.flags().contains(MemoryAttributeFlags::NEED_INITIATOR) {
                let (actual_initiators, _values) = attr.initiators(target).unwrap_or_default();
                if actual_initiators.is_empty() {
                    prop_oneof![
                        1 => Just(None),
                        4 => any_initiator(),
                    ]
                    .boxed()
                } else {
                    let actual_initiator = prop::sample::select(actual_initiators.clone());
                    prop_oneof![
                        1 => Just(None),
                        2 => actual_initiator.prop_flat_map(move |actual_initiator| {
                            match actual_initiator {
                                MemoryAttributeLocation::CpuSet(set) => {
                                    set_with_reference(set)
                                        .prop_map(|set| Some(OwnedAttributeLocation::CpuSet(set)))
                                        .boxed()
                                }
                                MemoryAttributeLocation::Object(obj) => {
                                    Just(Some(OwnedAttributeLocation::Object(obj)))
                                        .boxed()
                                }
                            }
                        }),
                        2 => any_initiator(),
                    ]
                    .boxed()
                }
            } else {
                prop_oneof![
                    2 => Just(None),
                    3 => any_initiator(),
                ]
                .boxed()
            };
            (Just(attr), Just(target), initiator)
        })
    }

    proptest! {
        #[test]
        fn value_query_errors(
            (attr, target, initiator) in memory_attribute_target_initiator()
        ) {
            // Avoid invalid value() calls which are not caught by hwloc's error
            // handling, resulting in segfaults until
            // https://github.com/open-mpi/hwloc/issues/685 is resolved.
            let locality_name = CString::new("Locality").unwrap();
            if attr.name() == locality_name.as_c_str() && target.cpuset().is_none() {
                return Ok(());
            }

            // Detect errors which are handled on the Rust side
            let foreign_initiator = matches!(initiator,
                Some(OwnedAttributeLocation::Object(obj)) if !attr.topology.contains(obj)
            );
            let foreign_target = !attr.topology.contains(target);
            let attr_needs_initiator = attr.flags().contains(MemoryAttributeFlags::NEED_INITIATOR);
            let need_initiator = attr_needs_initiator && initiator.is_none();
            let unwanted_initiator = !attr_needs_initiator && initiator.is_some();
            let any_error = foreign_initiator || foreign_target || need_initiator || unwanted_initiator;

            // Detect invalid (target, initiator) pairs.
            let (valid_targets, _values) = attr.targets(None).unwrap();
            let invalid_initiator_or_target = (attr_needs_initiator != initiator.is_some()) || valid_targets.iter().all(|candidate| {
                if candidate.global_persistent_index() != target.global_persistent_index() {
                    return true;
                }
                if !attr_needs_initiator {
                    return false;
                }
                let (initiators, _values) = attr.initiators(candidate).unwrap();
                initiators.iter().all(|candidate| match (candidate, &initiator) {
                    (MemoryAttributeLocation::CpuSet(candidate), Some(OwnedAttributeLocation::CpuSet(set))) => {
                        !candidate.includes(set)
                    }
                    (MemoryAttributeLocation::Object(candidate), Some(OwnedAttributeLocation::Object(obj))) => {
                        candidate.global_persistent_index() != obj.global_persistent_index()
                    },
                    _ => true,
                })
            });

            // Call value query and check result
            match attr.value(borrow_owned_initiator(&initiator), target) {
                Ok(value) => {
                    prop_assert!(!any_error);
                    // NOTE: Right now we're only testing this direction because
                    //       hwloc can yield attribute values for invalid
                    //       (initiator, target) pairs.
                    if value.is_none() {
                        prop_assert!(invalid_initiator_or_target);
                    }
                }
                Err(HybridError::Rust(er)) => match er {
                    ValueQueryError::BadInitiator(ei) => match ei {
                        InitiatorInputError::ForeignInitiator(_) => {
                            prop_assert!(foreign_initiator);
                            prop_assert!(matches!(
                                initiator,
                                Some(OwnedAttributeLocation::Object(obj))
                                if ei == InitiatorInputError::from(obj)
                            ));
                        }
                        InitiatorInputError::NeedInitiator(_) => prop_assert!(need_initiator),
                        InitiatorInputError::UnwantedInitiator(_) => prop_assert!(unwanted_initiator),
                    },
                    ValueQueryError::ForeignTarget(_) => prop_assert!(foreign_target),
                },
                Err(HybridError::Hwloc(e)) => unreachable!("Unexpected hwloc error {e}"),
            };
        }
    }

    /// Given a memory attribute, pick an initiator that has a chance of being
    /// correct for this attribute, but may be a random object possibly coming
    /// from another topology.
    fn memory_attribute_and_initiator(
    ) -> impl Strategy<Value = (MemoryAttribute<'static>, Option<OwnedAttributeLocation>)> {
        memory_attribute_target_initiator().prop_map(|(attr, _target, initiator)| (attr, initiator))
    }

    proptest! {
        #[test]
        fn target_query_errors(
            (attr, initiator) in memory_attribute_and_initiator()
        ) {
            let foreign_initiator = matches!(initiator,
                Some(OwnedAttributeLocation::Object(obj)) if !attr.topology.contains(obj)
            );
            let attr_needs_initiator = attr.flags().contains(MemoryAttributeFlags::NEED_INITIATOR);
            let need_initiator = attr_needs_initiator && initiator.is_none();
            let unwanted_initiator = !attr_needs_initiator && initiator.is_some();

            let check_normalized_error = |res_wo_ok: Result<(), InitiatorInputError>,
                                          initiator_is_optional: bool| {
                match res_wo_ok {
                    Ok(()) => prop_assert!(
                        !(foreign_initiator || unwanted_initiator)
                        && (initiator_is_optional || !need_initiator)
                    ),
                    Err(InitiatorInputError::ForeignInitiator(_)) => prop_assert!(foreign_initiator),
                    Err(InitiatorInputError::NeedInitiator(_)) => prop_assert!(
                        need_initiator && !initiator_is_optional
                    ),
                    Err(InitiatorInputError::UnwantedInitiator(_)) => prop_assert!(unwanted_initiator),
                }
                Ok(())
            };

            let initiator = borrow_owned_initiator(&initiator);
            check_normalized_error(
                attr.best_target(initiator).map(std::mem::drop),
                false,
            )?;
            check_normalized_error(
                attr.targets(initiator).map(std::mem::drop).map_err(|e| {
                    let HybridError::Rust(e) = e else {
                        unreachable!("No known error cases that aren't handled on the Rust side, yet got {e:?}")
                    };
                    e
                }),
                true,
            )?;
        }
    }

    /// Owned version of [`NUMALocation`]
    #[derive(Clone, Debug)]
    enum OwnedNUMALocation {
        Local {
            location: OwnedAttributeLocation,
            flags: LocalNUMANodeFlags,
        },
        All,
    }

    /// Generate an `OwnedNUMALocation`
    fn numa_location() -> impl Strategy<Value = OwnedNUMALocation> {
        let topology = Topology::test_instance();
        prop_oneof![
            1 => Just(OwnedNUMALocation::All),
            2 => {
                (any_object(), any::<LocalNUMANodeFlags>()).prop_map(|(obj, flags)| {
                    OwnedNUMALocation::Local {
                        location: OwnedAttributeLocation::Object(obj),
                        flags,
                    }
                })
            },
            2 => {
                let numa_nodes = topology.objects_with_type(ObjectType::NUMANode).filter(|numa| {
                    numa.cpuset().unwrap().weight().unwrap() > 0
                }).collect::<Vec<_>>();
                prop::sample::select(numa_nodes).prop_flat_map(|numa| {
                    (set_with_reference(numa.cpuset().unwrap()), any::<LocalNUMANodeFlags>()).prop_map(|(set, flags)| {
                        OwnedNUMALocation::Local {
                            location: OwnedAttributeLocation::CpuSet(set),
                            flags,
                        }
                    })
                })
            }
        ]
    }

    proptest! {
        #[test]
        fn local_numa_nodes(location in numa_location()) {
            let topology = Topology::test_instance();
            let location = match &location {
                OwnedNUMALocation::Local { location, flags } => {
                    let location = match location {
                        OwnedAttributeLocation::Object(obj) => MemoryAttributeLocation::from(*obj),
                        OwnedAttributeLocation::CpuSet(set) => MemoryAttributeLocation::from(set),
                    };
                    NUMALocation::Local { location, flags: *flags }
                }
                OwnedNUMALocation::All => NUMALocation::All,
            };
            let res = topology.local_numa_nodes(location);

            let foreign_object = if let NUMALocation::Local { location: MemoryAttributeLocation::Object(obj), .. } = location {
                !topology.contains(obj)
            } else {
                false
            };
            prop_assert_eq!(res.is_err(), foreign_object);
            let Ok(nodes) = res else { return Ok(()); };
            let node_ids = nodes.into_iter()
                                .map(TopologyObject::global_persistent_index)
                                .collect::<HashSet<_>>();

            match location {
                NUMALocation::All => {
                    let all_node_ids = topology.objects_with_type(ObjectType::NUMANode)
                                               .map(TopologyObject::global_persistent_index)
                                               .collect::<HashSet<_>>();
                    prop_assert_eq!(node_ids, all_node_ids);
                }
                NUMALocation::Local { location, flags } => {
                    let cpuset = match location {
                        MemoryAttributeLocation::CpuSet(set) => set,
                        MemoryAttributeLocation::Object(obj) => {
                            obj.cpuset()
                               .unwrap_or_else(|| obj.first_non_io_ancestor()
                                                     .unwrap()
                                                     .cpuset()
                                                     .unwrap())
                        }
                    };
                    let expected_node_ids = topology
                        .objects_with_type(ObjectType::NUMANode)
                        .filter_map(|node| {
                            let node_cpuset = node.cpuset().unwrap();
                            let flags = flags - LocalNUMANodeFlags::ALL;
                            let is_match = (flags == LocalNUMANodeFlags::empty() && node_cpuset == cpuset)
                                || (flags.contains(LocalNUMANodeFlags::LARGER_LOCALITY) && node_cpuset.includes(cpuset))
                                || (flags.contains(LocalNUMANodeFlags::SMALLER_LOCALITY) && cpuset.includes(node_cpuset));
                            is_match.then(|| node.global_persistent_index())
                        })
                        .collect::<HashSet<_>>();
                    prop_assert_eq!(expected_node_ids, node_ids);
                }
            }
        }
    }

    fn attr_names() -> &'static [String] {
        static NAMES: OnceLock<Vec<String>> = OnceLock::new();
        NAMES
            .get_or_init(|| {
                MemoryAttribute::all(Topology::test_instance())
                    .map(|attr| attr.name().to_str().unwrap().to_owned())
                    .collect::<Vec<_>>()
            })
            .as_slice()
    }

    // New memory attribute name with a fair chance of collision
    fn attribute_name() -> impl Strategy<Value = String> {
        prop_oneof![
            2 => prop::sample::select(attr_names()),
            3 => any_string(),
        ]
    }

    proptest! {
        #[test]
        fn build_empty_attribute(
            name in attribute_name(),
            flags: MemoryAttributeFlags
        ) {
            let initial_topology = Topology::test_instance();
            let mut topology = initial_topology.clone();
            let res = topology.edit(|editor| {
                editor
                    .register_memory_attribute(&name, flags)
                    .map(std::mem::drop)
            });

            let bad_flags =
                (flags & (MemoryAttributeFlags::HIGHER_IS_BEST
                          | MemoryAttributeFlags::LOWER_IS_BEST))
                    .iter()
                    .count() != 1;
            let name_contains_nul = name.contains('\0');
            let name_taken = attr_names().contains(&name);
            match res {
                Ok(()) => prop_assert!(!(bad_flags || name_contains_nul || name_taken)),
                Err(e) => {
                    match e {
                        RegisterError::BadFlags(bf) => {
                            prop_assert!(bad_flags, "Got unexpected BadFlags error with flags {flags:?}");
                            prop_assert_eq!(bf, FlagsError::from(flags));
                        }
                        RegisterError::NameContainsNul => prop_assert!(name_contains_nul),
                        RegisterError::NameTaken(taken) => {
                            prop_assert!(name_taken);
                            prop_assert_eq!(&*name, &*taken);
                        }
                    }
                    prop_assert_eq!(&topology, initial_topology);
                    return Ok(());
                }
            }
            prop_assert_ne!(&topology, initial_topology);

            let attr = topology.memory_attribute_named(&name).unwrap().unwrap();
            let cname = CString::new(name).unwrap();
            prop_assert_eq!(attr.name(), cname.as_c_str());
            prop_assert_eq!(attr.flags(), flags);
            let (targets, values) = attr.targets(None).unwrap();
            prop_assert!(targets.is_empty());
            prop_assert_eq!(values, None);

            let mut expected_dump = initial_topology.dump_memory_attributes();
            expected_dump.0.push(AttributeDump::new(attr));
            prop_assert!(topology.dump_memory_attributes().eq_modulo_topology(&expected_dump));
        }
    }

    // TODO: Build attribute with contents, test attribute dumps and their Debug

    // TODO: Check coverage and see if I need any other test
}
