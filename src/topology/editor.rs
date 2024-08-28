//! Modifying a loaded Topology
//!
//! In an ideal world, modifying a topology would just be a matter of calling
//! methods on an `&mut Topology`. Alas, this binding has to make it a little
//! more complicated than that due to the following reasons:
//!
//! - hwloc employs lazy caching patterns in such a way that after editing the
//!   topology, calling functions on an `*const hwloc_topology` may modify it
//!   in a thread-unsafe way. This is deeply at odds with the general design of
//!   the Rust aliasing model, and accounting for it by simply marking topology
//!   objects as internally mutable would result in major usability regressions
//!   (e.g. [`TopologyObject`] could not be [`Sync`]).
//! - Many hwloc topology editing functions take one or more `*const hwloc_obj`
//!   as a parameter. This is at odds with the simplest way to model topology
//!   object lookup in Rust, namely as borrows from the source [`Topology`],
//!   because once you have borrowed an `&TopologyObject` from a `&Topology`,
//!   you cannot call methods that require `&mut Topology` anymore. Working
//!   around this issue requires pointer-based unsafe code, carefully written
//!   so as not to violate Rust's aliasing model.
//! - While all of this would be workable through a sufficiently complicated API
//!   that lets the binding use internal mutability everywhere and delay
//!   creation of Rust references until the very moment where they are needed,
//!   one must bear in mind that topology editing is ultimately a niche feature
//!   which most hwloc users will never reach for. Common sense demands that it
//!   is the niche editing feature that takes an ergonomic and complexity hit,
//!   not the everyday topology queries.
//!
//! Therefore, topology editing is carried out using a dedicated
//! [`TopologyEditor`] type, defined in this module, which unfortunately has
//! sub-optimal ergonomics as a result of making the regular [`Topology`] type
//! as easy to use, cleanly implemented and feature-complete as it should be.

#[cfg(doc)]
use crate::topology::builder::{BuildFlags, TopologyBuilder};
use crate::{
    bitmap::{Bitmap, BitmapKind, BitmapRef, SpecializedBitmap, SpecializedBitmapRef},
    cpu::cpuset::CpuSet,
    errors::{self, ForeignObjectError, HybridError, NulError, ParameterError, RawHwlocError},
    ffi::{
        string::LibcString,
        transparent::{AsInner, AsNewtype},
    },
    memory::nodeset::NodeSet,
    object::{attributes::GroupAttributes, types::ObjectType, TopologyObject},
    topology::{builder::TypeFilter, Topology},
};
use bitflags::bitflags;
use errno::Errno;
use hwlocality_sys::{
    hwloc_restrict_flags_e, hwloc_topology, HWLOC_ALLOW_FLAG_ALL, HWLOC_ALLOW_FLAG_CUSTOM,
    HWLOC_ALLOW_FLAG_LOCAL_RESTRICTIONS, HWLOC_RESTRICT_FLAG_ADAPT_IO,
    HWLOC_RESTRICT_FLAG_ADAPT_MISC, HWLOC_RESTRICT_FLAG_BYNODESET,
    HWLOC_RESTRICT_FLAG_REMOVE_CPULESS, HWLOC_RESTRICT_FLAG_REMOVE_MEMLESS,
};
use libc::{EINVAL, ENOMEM, ENOSYS};
#[allow(unused)]
#[cfg(test)]
use similar_asserts::assert_eq;
use std::{
    fmt::{self, Debug, Write},
    panic::{AssertUnwindSafe, UnwindSafe},
    ptr::{self, NonNull},
};
use thiserror::Error;

/// # Modifying a loaded `Topology`
//
// --- Implementation details ---
//
// Upstream docs: https://hwloc.readthedocs.io/en/v2.9/group__hwlocality__tinker.html
impl Topology {
    /// Modify this topology
    ///
    /// hwloc employs lazy caching patterns that do not interact well with
    /// Rust's shared XOR mutable aliasing model. This API lets you safely
    /// modify the active `Topology` through a [`TopologyEditor`] proxy object,
    /// with the guarantee that by the time `Topology::edit()` returns, the
    /// `Topology` will be back in a state where it is safe to use `&self` again.
    ///
    /// In general, the hwlocality binding optimizes the ergonomics and
    /// performance of reading and using topologies at the expense of making
    /// them harder and slower to edit. If a strong need for easier or more
    /// efficient topology editing emerged, the right thing to do would
    /// probably be to set up an alternate hwloc Rust binding optimized for
    /// that, sharing as much code as possible with hwlocality.
    #[doc(alias = "hwloc_topology_refresh")]
    pub fn edit<R>(&mut self, edit: impl UnwindSafe + FnOnce(&mut TopologyEditor<'_>) -> R) -> R {
        // Set up topology editing
        let mut editor = TopologyEditor::new(self);
        let mut editor = AssertUnwindSafe(&mut editor);

        // Run the user-provided edit callback, catching panics
        let result = std::panic::catch_unwind(move || edit(&mut editor));

        // Force eager evaluation of all caches
        self.refresh();

        // Return user callback result or resume unwinding as appropriate
        match result {
            Ok(result) => result,
            Err(e) => std::panic::resume_unwind(e),
        }
    }

    /// Force eager evaluation of all lazily evaluated caches in preparation for
    /// using or exposing &self
    ///
    /// # Aborts
    ///
    /// A process abort will occur if this fails as we must not let an invalid
    /// `Topology` state escape, not even via unwinding, as that would result in
    /// undefined behavior (mutation which the compiler assumes will not happen).
    #[allow(clippy::print_stderr)]
    pub(crate) fn refresh(&mut self) {
        // Evaluate all the caches
        // SAFETY: - Topology is trusted to contain a valid ptr (type invariant)
        //         - hwloc ops are trusted to keep *mut parameters in a
        //           valid state unless stated otherwise
        let result = errors::call_hwloc_int_normal("hwloc_topology_refresh", || unsafe {
            hwlocality_sys::hwloc_topology_refresh(self.as_mut_ptr())
        });
        if let Err(e) = result {
            eprintln!("ERROR: Failed to refresh topology ({e}), so it's stuck in a state that violates Rust aliasing rules. Must abort...");
            std::process::abort()
        }

        // Check topology for correctness before exposing it
        if cfg!(debug_assertions) {
            // SAFETY: - Topology is trusted to contain a valid ptr (type invariant)
            //         - hwloc ops are trusted not to modify *const parameters
            unsafe { hwlocality_sys::hwloc_topology_check(self.as_ptr()) }
        }
    }
}

/// Proxy for modifying a `Topology`
///
/// This proxy object is carefully crafted to only allow operations that are
/// safe while modifying a topology and minimize the number of times the hwloc
/// lazy caches will need to be refreshed.
///
/// The API is broken down into sections roughly following the structure of the
/// upstream hwloc documentation:
///
/// - [General-purpose utilities](#general-purpose-utilities)
/// - [Basic modifications](#basic-modifications)
#[cfg_attr(
    feature = "hwloc-2_5_0",
    doc = "- [Add distances between objects](#add-distances-between-objects) (hwloc 2.5+)"
)]
/// - [Remove distances between objects](#remove-distances-between-objects)
/// - [Managing memory attributes](#managing-memory-attributes)
#[cfg_attr(
    feature = "hwloc-2_4_0",
    doc = "- [Kinds of CPU cores](#kinds-of-cpu-cores) (hwloc 2.4+)"
)]
//
// --- Implementation details
//
// Not all of the TopologyEditor API is implemented in the core editor.rs
// module. Instead, functionality which is very strongly related to one other
// code module is implemented in that module, leaving the editor module focused
// on basic lifecycle and cross-cutting issues.
#[derive(Debug)]
pub struct TopologyEditor<'topology>(&'topology mut Topology);

/// # General-purpose utilities
impl<'topology> TopologyEditor<'topology> {
    /// Wrap an `&mut Topology` into a topology editor
    pub(crate) fn new(topology: &'topology mut Topology) -> Self {
        Self(topology)
    }

    /// Get a shared reference to the inner Topology
    ///
    /// This requires rebuilding inner caches, which can be costly. Prefer
    /// accessing the topology before or after editing it if possible.
    pub fn topology(&mut self) -> &Topology {
        self.topology_mut().refresh();
        self.topology_mut()
    }

    /// Get a mutable reference to the inner Topology
    pub(crate) fn topology_mut(&mut self) -> &mut Topology {
        self.0
    }

    /// Contained hwloc topology pointer (for interaction with hwloc)
    pub(crate) fn topology_mut_ptr(&mut self) -> *mut hwloc_topology {
        self.topology_mut().as_mut_ptr()
    }
}

/// # Basic modifications
//
// --- Implementation details ---
//
// Upstream docs: https://hwloc.readthedocs.io/en/v2.9/group__hwlocality__tinker.html
impl<'topology> TopologyEditor<'topology> {
    /// Restrict the topology to the given CPU set or nodeset
    ///
    /// The topology is modified so as to remove all objects that are not
    /// included (or partially included) in the specified [`CpuSet`] or
    /// [`NodeSet`] set. All objects CPU and node sets are restricted
    /// accordingly.
    ///
    /// Restricting the topology removes some locality information, hence the
    /// remaining objects may get reordered (including PUs and NUMA nodes), and
    /// their logical indices may change.
    ///
    /// This call may not be reverted by restricting back to a larger set. Once
    /// dropped during restriction, objects may not be brought back, except by
    /// loading another topology with [`Topology::new()`] or [`TopologyBuilder`].
    ///
    /// # Errors
    ///
    /// It is an error to attempt to remove all CPUs or NUMA nodes from a
    /// topology using a `set` that has no intersection with the relevant
    /// topology set. The topology will not be modified in this case, and a
    /// [`ParameterError`] will be returned instead.
    ///
    /// # Aborts
    ///
    /// Failure to allocate internal data will lead to a process abort, because
    /// the topology gets corrupted in this case and must not be touched again,
    /// but we have no way to prevent this in a safe API.
    #[allow(clippy::print_stderr)]
    #[doc(alias = "hwloc_topology_restrict")]
    pub fn restrict<SetRef: SpecializedBitmapRef>(
        &mut self,
        set: SetRef,
        flags: RestrictFlags,
    ) -> Result<(), ParameterError<SetRef::Owned>> {
        /// Polymorphized version of this function (avoids generics code bloat)
        fn polymorphized<OwnedSet: SpecializedBitmap>(
            self_: &mut TopologyEditor<'_>,
            set: &OwnedSet,
            mut flags: RestrictFlags,
        ) -> Result<(), ParameterError<OwnedSet>> {
            // Check if applying this restriction would remove all CPUs/nodes
            //
            // This duplicates some error handling logic inside of hwloc, but
            // reduces the odds that in the presence of errno reporting issues
            // on Windows, the process will abort when it shouldn't.
            let topology = self_.topology();
            let erased_set = set.as_bitmap_ref();
            let (affected, other) = match OwnedSet::BITMAP_KIND {
                BitmapKind::CpuSet => {
                    let topology_set = topology.cpuset();
                    let topology_set: &Bitmap = topology_set.as_ref();
                    let cpuset = CpuSet::from(erased_set & topology_set);
                    let nodeset = NodeSet::from_cpuset(topology, &cpuset);
                    (Bitmap::from(cpuset), Bitmap::from(nodeset))
                }
                BitmapKind::NodeSet => {
                    let topology_set = topology.nodeset();
                    let topology_set: &Bitmap = topology_set.as_ref();
                    let nodeset = NodeSet::from(erased_set & topology_set);
                    let cpuset = CpuSet::from_nodeset(topology, &nodeset);
                    (Bitmap::from(nodeset), Bitmap::from(cpuset))
                }
            };
            if affected.is_empty()
                && (flags.contains(RestrictFlags::REMOVE_EMPTIED) || other.is_empty())
            {
                return Err(ParameterError::from(set.to_owned()));
            }

            // Configure restrict flags correctly depending on the node set type
            match OwnedSet::BITMAP_KIND {
                BitmapKind::CpuSet => flags.remove(RestrictFlags::BY_NODE_SET),
                BitmapKind::NodeSet => flags.insert(RestrictFlags::BY_NODE_SET),
            }
            flags.remove(RestrictFlags::REMOVE_CPULESS | RestrictFlags::REMOVE_MEMLESS);
            if flags.contains(RestrictFlags::REMOVE_EMPTIED) {
                flags.remove(RestrictFlags::REMOVE_EMPTIED);
                match OwnedSet::BITMAP_KIND {
                    BitmapKind::CpuSet => {
                        flags.insert(RestrictFlags::REMOVE_CPULESS);
                    }
                    BitmapKind::NodeSet => {
                        flags.insert(RestrictFlags::REMOVE_MEMLESS);
                    }
                }
            }

            // Apply requested restriction
            // SAFETY: - Topology is trusted to contain a valid ptr (type invariant)
            //         - hwloc ops are trusted to keep *mut parameters in a
            //           valid state unless stated otherwise
            //         - set trusted to be valid (Bitmap type invariant)
            //         - hwloc ops are trusted not to modify *const parameters
            //         - By construction, only allowed flag combinations may be sent
            //           to hwloc
            let result = errors::call_hwloc_int_normal("hwloc_topology_restrict", || unsafe {
                hwlocality_sys::hwloc_topology_restrict(
                    self_.topology_mut_ptr(),
                    set.as_bitmap_ref().as_ptr(),
                    flags.bits(),
                )
            });
            let handle_enomem = |certain: bool| {
                let nuance = if certain { "is" } else { "might be" };
                eprintln!("ERROR: Topology {nuance} stuck in an invalid state. Must abort...");
                std::process::abort()
            };
            match result {
                Ok(_) => Ok(()),
                Err(
                    raw_err @ RawHwlocError {
                        errno: Some(errno), ..
                    },
                ) => match errno.0 {
                    EINVAL => Err(ParameterError::from(set.to_owned())),
                    ENOMEM => handle_enomem(true),
                    _ => unreachable!("Unexpected hwloc error: {raw_err}"),
                },
                Err(raw_err @ RawHwlocError { errno: None, .. }) => {
                    if cfg!(windows) {
                        // Due to errno propagation issues on windows, we may not
                        // know which of EINVAL and ENOMEM we're dealing with. Since
                        // not aborting on ENOMEM is unsafe, we must take the
                        // pessimistic assumption that it was ENOMEM and abort...
                        handle_enomem(false)
                    } else {
                        unreachable!("Unexpected hwloc error: {raw_err}")
                    }
                }
            }
        }
        polymorphized(self, &*set, flags)
    }

    /// Change the sets of allowed PUs and NUMA nodes in the topology
    ///
    /// This function only works if [`BuildFlags::INCLUDE_DISALLOWED`] was set
    /// during topology building. It does not modify any object, it only changes
    /// the sets returned by [`Topology::allowed_cpuset()`] and
    /// [`Topology::allowed_nodeset()`].
    ///
    /// It is notably useful when importing a topology from another process
    /// running in a different Linux Cgroup.
    ///
    /// Removing objects from a topology should rather be performed with
    /// [`TopologyEditor::restrict()`].
    ///
    /// # Errors
    ///
    /// - [`EmptyCustom`] if an `AllowSet::Custom` does not do anything because
    ///   both its `cpuset` and `nodeset` members are empty.
    /// - [`InvalidCpuset`] if applying the `cpuset` of an `AllowSet::Custom`
    ///   would amount to disallowing all CPUs from the topology.
    /// - [`InvalidNodeset`] if applying the `nodeset` of an `AllowSet::Custom`
    ///   would amount to disallowing all NUMA nodes from the topology.
    /// - [`Unsupported`] if the specified `AllowSet` is not supported by the
    ///   host operating system.
    ///
    /// [`EmptyCustom`]: AllowSetError::EmptyCustom
    /// [`InvalidCpuset`]: AllowSetError::InvalidCpuset
    /// [`InvalidNodeset`]: AllowSetError::InvalidNodeset
    /// [`Unsupported`]: AllowSetError::Unsupported
    #[doc(alias = "hwloc_topology_allow")]
    pub fn allow(&mut self, allow_set: AllowSet<'_>) -> Result<(), HybridError<AllowSetError>> {
        // Convert AllowSet into a valid `hwloc_topology_allow` configuration
        let (cpuset, nodeset, flags) = match allow_set {
            AllowSet::All => (ptr::null(), ptr::null(), HWLOC_ALLOW_FLAG_ALL),
            AllowSet::LocalRestrictions => (
                ptr::null(),
                ptr::null(),
                HWLOC_ALLOW_FLAG_LOCAL_RESTRICTIONS,
            ),
            AllowSet::Custom { cpuset, nodeset } => {
                // Check that this operation does not empty any allow-set
                let topology = self.topology();
                if let Some(cpuset) = cpuset {
                    if !topology.cpuset().intersects(cpuset) {
                        return Err(AllowSetError::InvalidCpuset.into());
                    }
                }
                if let Some(nodeset) = nodeset {
                    if !topology.nodeset().intersects(nodeset) {
                        return Err(AllowSetError::InvalidNodeset.into());
                    }
                }

                // Check that at least one set has been specified
                let cpuset = cpuset.map_or(ptr::null(), CpuSet::as_ptr);
                let nodeset = nodeset.map_or(ptr::null(), NodeSet::as_ptr);
                if cpuset.is_null() && nodeset.is_null() {
                    return Err(AllowSetError::EmptyCustom.into());
                }
                (cpuset, nodeset, HWLOC_ALLOW_FLAG_CUSTOM)
            }
        };

        // Call hwloc
        // SAFETY: - Topology is trusted to contain a valid ptr (type invariant)
        //         - hwloc ops are trusted to keep *mut parameters in a
        //           valid state unless stated otherwise
        //         - cpusets and nodesets are trusted to be valid (type invariant)
        //         - hwloc ops are trusted not to modify *const parameters
        //         - By construction, flags are trusted to be in sync with the
        //           cpuset and nodeset params + only one of them is set as
        //           requested by hwloc
        let result = errors::call_hwloc_int_normal("hwloc_topology_allow", || unsafe {
            hwlocality_sys::hwloc_topology_allow(self.topology_mut_ptr(), cpuset, nodeset, flags)
        });
        match result {
            Ok(_) => Ok(()),
            Err(RawHwlocError {
                errno: Some(Errno(ENOSYS)),
                ..
            }) => Err(AllowSetError::Unsupported.into()),
            Err(other) => Err(HybridError::Hwloc(other)),
        }
    }

    /// Add more structure to the topology by creating an intermediate [`Group`]
    ///
    /// Sibling normal objects below a common parent object can be grouped to
    /// express that there is a resource shared between the underlying CPU
    /// cores, which cannot be modeled using a more specific standard hwloc
    /// object type. For example, this is how the intra-chip NUMA clusters of
    /// modern high-core-count AMD and Intel CPUs are usually modeled. See the
    /// ["What are these Group objects in my
    /// topology"](https://hwloc.readthedocs.io/en/v2.9/faq.html#faq_groups)
    /// entry of the hwloc FAQ for more information.
    ///
    /// Alas, creating hwloc groups is a lot less straightforward than the above
    /// summary may suggest, and you are strongly advised to carefully read and
    /// understand all of the following before using this function.
    ///
    ///
    /// # Group creation guide
    ///
    /// ## Basic workflow
    ///
    /// This function will first call the `find_parent` callback in order to
    /// identify the parent object under which a new group should be inserted.
    ///
    /// The callback(s) specified by `child_filter` will then be called on
    /// each normal and/or memory child of this parent, allowing you to tell
    /// which objects should become members of the newly created group. See
    /// [`GroupChildFilter`] for more information.
    ///
    /// This API design, which may be unexpectedly complex, helps you honor
    /// hwloc's many group creation rules:
    ///
    /// - Only normal and memory objects can be members of a group. I/O and
    ///   [`Misc`] objects can only be grouped coarsely and indirectly by
    ///   grouping the normal objects under which they reside.
    /// - The normal and memory members of an hwloc group must be consistent
    ///   with each other, as explained in the [`GroupChildFilter`]
    ///   documentation.
    /// - It is, generally speaking, not possible to group objects which do not
    ///   lie below the same parent. For example, you cannot create a group that
    ///   contains the first hyperthreads of each core of an x86 CPU.
    ///
    /// One extra constraint that **you** are responsible for honoring is that
    /// hwloc does not support empty groups. Therefore your `child_filter`
    /// callback(s) must select at least one normal or memory child.
    ///
    /// Finally, the `dont_merge` parameter allows you to adjust hwloc's
    /// strategy for merging proposed groups with equivalent topology objects,
    /// as explained in the following section.
    ///
    /// ## Equivalence and merging
    ///
    /// hwloc considers a group to be equivalent to one or more existing
    /// topology objects in the following circumstances:
    ///
    /// * A group with a single child object is considered to be equivalent to
    ///   this child object
    /// * A group which covers all children of the parent object that was
    ///   designated by `find_parent` is considered to be equivalent to this
    ///   parent object
    ///     - This typically happens as a result of your children selection
    ///       callbacks returning `true` for all children of the parent object.
    ///     - If you were using [`GroupChildFilter::Mixed`] with `strict` set to
    ///       `false`, it may also happen that although one of your callbacks
    ///       did not pick all children, the remaining children had to be added
    ///       to follow hwloc's group consistency rules.
    ///
    /// In addition to these equivalence relations, topology objects which form
    /// a single-child chain with identical cpusets and nodesets (a simple
    /// example being L2 -> L1d -> L1i -> Core chains in x86 topologies), are
    /// also considered to be equivalent to each other. Therefore, if a group is
    /// considered to be equivalent to one of these objects, then it is
    /// considered equivalent to all of them.
    ///
    /// When a proposed group is equivalent to an existing topology object, the
    /// default hwloc behavior is not to create a group, but instead to return
    /// [`InsertedGroup::Existing`] with one of the objects that is considered
    /// equivalent to the proposed group as a parameter. The idea is that you do
    /// not really need a group to model the desired set of CPU cores and NUMA
    /// nodes, since at least one existing topology object already does so.
    ///
    /// If you want to force the creation of a group in a situation where hwloc
    /// would not create one, you can set `dont_merge` to `true` to force the
    /// creation of a group even when hwloc considers the proposed group to be
    /// equivalent to one existing topology object. This comes with two caveats:
    ///
    /// - The group may be created above or below any of the objects that it is
    ///   considered equivalent to, not necessarily below the parent object that
    ///   you initially had in mind.
    /// - Even with this option, hwloc will refuse to create a group that is
    ///   equivalent to the topology root.
    ///
    /// ## Documenting groups
    ///
    /// By nature, the [`Group`] object type is not very descriptive of what the
    /// group represents in hardware, so you may want to add extra annotations
    /// describing what the group is about.
    ///
    /// To this end, after a successful group object insertion, you may use
    #[cfg_attr(windows, doc = "[`TopologyObject::set_subtype_unchecked()`]")]
    #[cfg_attr(not(windows), doc = "[`TopologyObject::set_subtype()`]")]
    /// to have `lstopo` display something other than "Group" as the type name.
    ///
    /// If needed, you can also complement this basic group type information
    /// with any number of extra name/value info pairs you need using
    /// [`TopologyObject::add_info()`].
    ///
    /// ## Identifier invalidation
    ///
    /// When a group is created, it becomes a child of the group members' former
    /// parent. To allow for this, the normal children of this parent need to be
    /// reordered first, so that the group members lie at consecutive indices. A
    /// new depth level of type [`Group`] may also need to be created to host
    /// the group, which will push existing depths downwards. As a consequence
    /// of all these topology changes...
    ///
    /// - The logical indices of all objects at the depth where the group
    ///   members used to lie may change as a result of calling this function.
    ///   If you want to identify a child object across calls to this function,
    ///   you should therefore use another identifier than the logical index or
    ///   sibling rank. [Global persistent
    ///   indices](TopologyObject::global_persistent_index()) are explicitly
    ///   designed for this use case.
    /// - The mapping of depths to object types may change as a result of
    ///   calling this function, for all depths below the designated group
    ///   parent. Therefore, you must be very cautious about reusing previously
    ///   computed depth values across calls to this function.
    ///
    ///
    /// # Errors
    ///
    /// - [`FilteredOut`] if one attempts to create a group in a topology where
    ///   they are filtered out using [`TypeFilter::KeepNone`].
    /// - [`BadParentType`] if the designated group parent is not a normal
    ///   object.
    /// - [`ForeignParent`] if the designated group parent does not belong to
    ///   the topology that is being edited.
    /// - [`Empty`] if the [`GroupChildFilter`] did not select any child.
    /// - [`Inconsistent`] if [`GroupChildFilter::Mixed`] was used in strict
    ///   mode, but the selected normal and memory object sets were not
    ///   consistent.
    ///
    /// [`BadParentType`]: InsertGroupError::BadParentType
    /// [`Empty`]: InsertGroupError::Empty
    /// [`FilteredOut`]: InsertGroupError::FilteredOut
    /// [`ForeignParent`]: InsertGroupError::ForeignParent
    /// [`Group`]: ObjectType::Group
    /// [`Inconsistent`]: InsertGroupError::Inconsistent
    /// [`Misc`]: ObjectType::Misc
    //
    // --- Implementation details ---
    //
    // In the future, find_children will be an impl FnOnce(&Topology) -> impl
    // IntoIterator<Item = &TopologyObject>, but impl Trait inside of impl
    // Trait is not allowed yet.
    #[doc(alias = "hwloc_topology_alloc_group_object")]
    #[doc(alias = "hwloc_obj_add_other_obj_sets")]
    #[doc(alias = "hwloc_topology_insert_group_object")]
    pub fn insert_group_object<NormalFilter, MemoryFilter>(
        &mut self,
        dont_merge: bool,
        find_parent: impl FnOnce(&Topology) -> &TopologyObject,
        child_filter: GroupChildFilter<NormalFilter, MemoryFilter>,
    ) -> Result<InsertedGroup<'topology>, HybridError<InsertGroupError>>
    where
        NormalFilter: FnMut(&TopologyObject) -> bool,
        MemoryFilter: FnMut(&TopologyObject) -> bool,
    {
        // Check type filter
        let group_filter = self
            .topology()
            .type_filter(ObjectType::Group)
            .map_err(HybridError::Hwloc)?;
        if group_filter == TypeFilter::KeepNone {
            return Err(InsertGroupError::FilteredOut.into());
        }

        // Create group object
        let mut group = AllocatedGroup::new(self).map_err(HybridError::Hwloc)?;
        group.add_children(find_parent, child_filter)?;
        group.configure_merging(dont_merge);
        group.insert().map_err(HybridError::Hwloc)
    }

    /// Add a [`Misc`] object as a leaf of the topology
    ///
    /// A new [`Misc`] object will be created and inserted into the topology as
    /// a child of the node selected by `find_parent`. It is appended to the
    /// list of existing Misc children, without ever adding any intermediate
    /// hierarchy level. This is useful for annotating the topology without
    /// actually changing the hierarchy.
    ///
    /// `name` is supposed to be unique across all [`Misc`] objects in the
    /// topology. It must not contain any NUL chars. If it contains any other
    /// non-printable characters, then they will be dropped when exporting to
    /// XML.
    ///
    /// The new leaf object will not have any cpuset.
    ///
    /// # Errors
    ///
    /// - [`FilteredOut`] if one attempts to create a Misc object in a topology
    ///   where they are filtered out using [`TypeFilter::KeepNone`].
    /// - [`ForeignParent`] if the parent `&TopologyObject` returned by
    ///   `find_parent` does not belong to this [`Topology`].
    /// - [`NameContainsNul`] if `name` contains NUL chars.
    /// - [`NameAlreadyExists`] if a Misc object called `name` already exists.
    ///
    /// [`FilteredOut`]: InsertMiscError::FilteredOut
    /// [`ForeignParent`]: InsertMiscError::ForeignParent
    /// [`Misc`]: ObjectType::Misc
    /// [`NameAlreadyExists`]: InsertMiscError::NameAlreadyExists
    /// [`NameContainsNul`]: InsertMiscError::NameContainsNul
    #[doc(alias = "hwloc_topology_insert_misc_object")]
    pub fn insert_misc_object(
        &mut self,
        name: &str,
        find_parent: impl FnOnce(&Topology) -> &TopologyObject,
    ) -> Result<&'topology mut TopologyObject, HybridError<InsertMiscError>> {
        /// Polymorphized version of this function (avoids generics code bloat)
        ///
        /// # Safety
        ///
        /// - `parent` must point to a [`TopologyObject`] that belongs to
        ///   `self_`
        /// - Any `&TopologyObject` that the pointer parent has been generated
        ///   from must be dropped before calling this function: we'll modify
        ///   its target, so reusing it would be UB.
        unsafe fn polymorphized<'topology>(
            self_: &mut TopologyEditor<'topology>,
            name: &str,
            parent: NonNull<TopologyObject>,
        ) -> Result<&'topology mut TopologyObject, HybridError<InsertMiscError>> {
            // Convert object name to a C string
            let name = LibcString::new(name)
                .map_err(|_| HybridError::Rust(InsertMiscError::NameContainsNul))?;

            // Call hwloc entry point
            let mut ptr =
                // SAFETY: - Topology is trusted to contain a valid ptr (type
                //           invariant)
                //         - hwloc ops are trusted to keep *mut parameters in a
                //           valid state unless stated otherwise
                //         - LibcString should yield valid C strings, which
                //           we're not using beyond their intended lifetime
                //         - hwloc ops are trusted not to modify *const
                //           parameters
                //         - Per polymorphized safety constract, parent should
                //           be correct and not be associated with a live &-ref
                errors::call_hwloc_ptr_mut("hwloc_topology_insert_misc_object", || unsafe {
                    hwlocality_sys::hwloc_topology_insert_misc_object(
                        self_.topology_mut_ptr(),
                        parent.as_inner().as_ptr(),
                        name.borrow(),
                    )
                })
                .map_err(HybridError::Hwloc)?;
            // SAFETY: - If hwloc succeeded, the output pointer is assumed to be
            //           valid and to point to a valid object
            //         - Output lifetime is bound to the topology that it comes
            //           from
            Ok(unsafe { ptr.as_mut().as_newtype() })
        }

        // Check type filter
        let topology = self.topology();
        let group_filter = topology
            .type_filter(ObjectType::Misc)
            .map_err(HybridError::Hwloc)?;
        if group_filter == TypeFilter::KeepNone {
            return Err(InsertMiscError::FilteredOut.into());
        }

        // Make sure no Misc object with this name exists
        if topology.objects_with_type(ObjectType::Misc).any(|obj| {
            let Some(obj_name) = obj.name() else {
                return false;
            };
            let Ok(obj_name) = obj_name.to_str() else {
                return false;
            };
            obj_name == name
        }) {
            return Err(InsertMiscError::NameAlreadyExists.into());
        }

        // Find parent object
        let parent: NonNull<TopologyObject> = {
            let parent = find_parent(topology);
            if !topology.contains(parent) {
                return Err(InsertMiscError::ForeignParent(parent.into()).into());
            }
            parent.into()
        };

        // SAFETY: parent comes from this topology, source ref has been dropped
        unsafe { polymorphized(self, name, parent) }
    }
}

bitflags! {
    /// Flags to be given to [`TopologyEditor::restrict()`]
    #[derive(Copy, Clone, Debug, Default, Eq, Hash, PartialEq)]
    #[doc(alias = "hwloc_restrict_flags_e")]
    pub struct RestrictFlags: hwloc_restrict_flags_e {
        /// Remove all objects that lost all resources of the target type
        ///
        /// By default, only objects that contain no PU and no memory are
        /// removed. This flag allows you to remove all objects that...
        ///
        /// - Do not have access to any CPU anymore when restricting by CpuSet
        /// - Do not have access to any memory anymore when restricting by NodeSet
        //
        // --- Implementation details ---
        //
        // This is a virtual flag that is cleared and mapped into
        // `REMOVE_CPULESS` or `REMOVE_MEMLESS` as appropriate.
        #[doc(alias = "HWLOC_RESTRICT_FLAG_REMOVE_CPULESS")]
        #[doc(alias = "HWLOC_RESTRICT_FLAG_REMOVE_MEMLESS")]
        const REMOVE_EMPTIED = hwloc_restrict_flags_e::MAX;

        /// Remove all objects that became CPU-less
        //
        // --- Implementation details ---
        //
        // This is what `REMOVE_EMPTIED` maps into when restricting by `CpuSet`.
        #[doc(hidden)]
        const REMOVE_CPULESS = HWLOC_RESTRICT_FLAG_REMOVE_CPULESS;

        /// Restrict by NodeSet insted of by `CpuSet`
        //
        // --- Implementation details ---
        //
        // This flag is automatically set when restricting by `NodeSet`.
        #[doc(hidden)]
        const BY_NODE_SET = HWLOC_RESTRICT_FLAG_BYNODESET;

        /// Remove all objects that became memory-less
        //
        // --- Implementation details ---
        //
        // This is what `REMOVE_EMPTIED` maps into when restricting by `NodeSet`.
        #[doc(hidden)]
        const REMOVE_MEMLESS = HWLOC_RESTRICT_FLAG_REMOVE_MEMLESS;

        /// Move Misc objects to ancestors if their parents are removed during
        /// restriction
        ///
        /// If this flag is not set, Misc objects are removed when their parents
        /// are removed.
        #[doc(alias = "HWLOC_RESTRICT_FLAG_ADAPT_MISC")]
        const ADAPT_MISC = HWLOC_RESTRICT_FLAG_ADAPT_MISC;

        /// Move I/O objects to ancestors if their parents are removed
        /// during restriction
        ///
        /// If this flag is not set, I/O devices and bridges are removed when
        /// their parents are removed.
        #[doc(alias = "HWLOC_RESTRICT_FLAG_ADAPT_IO")]
        const ADAPT_IO = HWLOC_RESTRICT_FLAG_ADAPT_IO;
    }
}
//
crate::impl_arbitrary_for_bitflags!(RestrictFlags, hwloc_restrict_flags_e);

/// Requested adjustment to the allowed set of PUs and NUMA nodes
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[doc(alias = "hwloc_allow_flags_e")]
pub enum AllowSet<'set> {
    /// Mark all objects as allowed in the topology
    #[doc(alias = "HWLOC_ALLOW_FLAG_ALL")]
    All,

    /// Only allow objects that are available to the current process
    ///
    /// Requires [`BuildFlags::ASSUME_THIS_SYSTEM`] so that the set of available
    /// resources can actually be retrieved from the operating system.
    #[doc(alias = "HWLOC_ALLOW_FLAG_LOCAL_RESTRICTIONS")]
    LocalRestrictions,

    /// Allow a custom set of objects
    ///
    /// You should provide at least one of `cpuset` and `nodeset`.
    ///
    /// No attempt is made to keep the allowed cpusets and nodesets consistent
    /// with each other, so you can end up in situations where e.g. access to
    /// some CPU cores is theoretically allowed by the topology's allowed
    /// cpuset, but actually prevented because their NUMA node is not part of
    /// the topology's allowed nodeset.
    #[doc(alias = "HWLOC_ALLOW_FLAG_CUSTOM")]
    Custom {
        /// New value of [`Topology::allowed_cpuset()`]
        cpuset: Option<&'set CpuSet>,

        /// New value of [`Topology::allowed_nodeset()`]
        nodeset: Option<&'set NodeSet>,
    },
}
//
impl fmt::Display for AllowSet<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AllowSet::Custom { cpuset, nodeset } => {
                let mut s = String::from("Custom(");
                if let Some(cpuset) = cpuset {
                    write!(s, "{cpuset}")?;
                    if nodeset.is_some() {
                        s.push_str(", ");
                    }
                }
                if let Some(nodeset) = nodeset {
                    write!(s, "{nodeset}")?;
                }
                s.push(')');
                f.pad(&s)
            }
            other @ (AllowSet::All | AllowSet::LocalRestrictions) => {
                <Self as fmt::Debug>::fmt(other, f)
            }
        }
    }
}
//
impl<'set> From<&'set CpuSet> for AllowSet<'set> {
    fn from(set: &'set CpuSet) -> Self {
        Self::Custom {
            cpuset: Some(set),
            nodeset: None,
        }
    }
}
//
impl<'set> From<&'set NodeSet> for AllowSet<'set> {
    fn from(set: &'set NodeSet) -> Self {
        Self::Custom {
            cpuset: None,
            nodeset: Some(set),
        }
    }
}

/// Error while trying to set the allow-set of a topology
#[derive(Copy, Clone, Debug, Eq, Error, Hash, PartialEq)]
pub enum AllowSetError {
    /// [`AllowSet::Custom`] was specified but both the `cpuset` and `nodeset`
    /// were empty, so it isn't clear how the allow set should change
    #[error("AllowSet::Custom cannot have both empty cpuset AND nodeset members")]
    EmptyCustom,

    /// [`AllowSet::Custom`] was specified with a cpuset that would disallow all
    /// CPUs from the topology
    #[error("AllowSet::Custom cannot be used to clear the topology's allowed cpuset")]
    InvalidCpuset,

    /// [`AllowSet::Custom`] was specified with a nodeset that would disallow
    /// all NUMA nodes from the topology
    #[error("AllowSet::Custom cannot be used to clear the topology's allowed nodeset")]
    InvalidNodeset,

    /// An unsupported [`AllowSet`] was passed in
    ///
    /// At the time of writing (2024-01-08), this happens when using
    /// [`AllowSet::LocalRestrictions`] on any operating system other than Linux
    /// and Solaris.
    #[error("this operation is not supported on this OS")]
    Unsupported,
}

/// Callbacks that selects the members of a proposed group object
///
/// The basic workflow of [`TopologyEditor::insert_group_object()`] is that you
/// first specify which topology object should be the parent of the newly
/// created group, and then you specify (using this enum and its inner
/// callbacks) which of the normal and memory children of this parent object
/// should become members of the newly created group.
///
/// However, as an extra complication, you must live with the fact that hwloc
/// only supports groups whose normal and memory member lists follow the
/// following consistency rules:
///
/// 1. If a memory object is a member of a group, then all normal objects which
///    are attached to this memory object (as evidenced by their PUs being part
///    of that memory object's cpuset) must also be members of this group.
/// 2. Conversely, if all normal objects which are attached to a memory object
///    are members of a group, then this memory object must also be made a
///    member of this group.
///
/// Because following these rules by hand is unpleasant, we provide various
/// shortcuts which allow you to only specify a subset of the group's members,
/// and let the remaining members required to follow the consistency rules be
/// added to the group automatically.
#[derive(Copy, Clone)]
pub enum GroupChildFilter<
    NormalFilter = fn(&TopologyObject) -> bool,
    MemoryFilter = fn(&TopologyObject) -> bool,
> where
    NormalFilter: FnMut(&TopologyObject) -> bool,
    MemoryFilter: FnMut(&TopologyObject) -> bool,
{
    /// Pick the group's normal children in the parent's normal children list
    ///
    /// Each normal child of the designated parent will be passed to the
    /// provided callback, which will tell if this child should be made a member
    /// of the group (`true`) or not (`false`), as in [`Iterator::filter()`].
    ///
    /// Memory children will then be automatically added in order to produce a
    /// group member set that follows the consistency rules.
    ///
    /// Due to a limitation of the Rust compiler, as of Rust 1.75 this type
    /// constructor mistakenly requires you to specify a `MemoryFilter` type
    /// parameter. You can work around this by using the [`Self::normal()`]
    /// constructor instead.
    Normal(NormalFilter),

    /// Pick the group's memory children in the parent's memory children list
    ///
    /// Works like `Normal`, except the provided filter is used to select memory
    /// children instead of normal children, and it is normal children that get
    /// automatically added to follow the consistency rules.
    ///
    /// Due to a limitation of the Rust compiler, as of Rust 1.75 this type
    /// constructor mistakenly requires you to specify a `NormalFilter` type
    /// parameter. You can work around this by using the [`Self::memory()`]
    /// constructor instead.
    Memory(MemoryFilter),

    /// Pick the group's normal and memory children
    ///
    /// The normal **and** memory children of the designated parent are
    /// traversed and filtered using the `normal` and `memory` filters
    /// respectively, as in the `Normal` and `Memory` cases.
    ///
    /// The resulting normal and memory children sets may or may not be
    /// subsequently expanded to follow the consistency rules, depending on the
    /// value of the `strict` flag.
    Mixed {
        /// Error out when `normal` and `memory` don't pick consistent sets
        ///
        /// If this flag isn't set, then after the `normal` and `memory`
        /// callbacks have picked preliminary normal and memory children lists,
        /// these normal and memory children lists are automatically expanded to
        /// honor the consistency rules. This gives you the smallest valid hwloc
        /// group that contains **at least** the children you asked for, at the
        /// cost of possibly getting extra children that you did not expect,
        /// which your code must handle gracefully.
        ///
        /// If this flag is set, then you are responsible for picking normal and
        /// memory children sets that honor the consistency rules, and
        /// [`TopologyEditor::insert_group_object()`] will fail if you don't.
        /// This is for situations where getting unexpected extra group members
        /// is unacceptable, and you are ready to go through the burden of
        /// applying the consistency rules yourself in order to avoid this
        /// outcome.
        strict: bool,

        /// Filter that selects the future group's normal children amongst the
        /// parent's normal children list, as in `Normal`
        normal: NormalFilter,

        /// Filter that selects the future group's memory children amongst the
        /// parent's memory children list, as in `Memory`
        memory: MemoryFilter,
    },
}
//
impl<NormalFilter> GroupChildFilter<NormalFilter, fn(&TopologyObject) -> bool>
where
    NormalFilter: FnMut(&TopologyObject) -> bool,
{
    /// Workaround for lack of default type parameter fallback when using the
    /// [`Self::Normal`] type constructor
    pub fn normal(filter: NormalFilter) -> Self {
        Self::Normal(filter)
    }
}
//
impl<MemoryFilter> GroupChildFilter<fn(&TopologyObject) -> bool, MemoryFilter>
where
    MemoryFilter: FnMut(&TopologyObject) -> bool,
{
    /// Workaround for lack of default type parameter fallback when using the
    /// [`Self::Memory`] type constructor
    pub fn memory(filter: MemoryFilter) -> Self {
        Self::Memory(filter)
    }
}
//
impl<NormalFilter, MemoryFilter> GroupChildFilter<NormalFilter, MemoryFilter>
where
    NormalFilter: FnMut(&TopologyObject) -> bool,
    MemoryFilter: FnMut(&TopologyObject) -> bool,
{
    /// Pick children of a group's parent according to this filter
    ///
    /// The group consistency rules given above actually describe the behavior
    /// of `hwloc_topology_insert_group_object()`. Because this API is
    /// cpuset/nodeset-based, you cannot add a NUMA node to a group without
    /// adding all the associated normal objects (since the normal objects are
    /// part of the NUMA node's cpuset), and you cannot add all of a NUMA node's
    /// CPUs without adding the NUMA node (since adding all NUMA node children
    /// sets all bits from the NUMA node's cpuset and nodeset).
    ///
    /// So if the group child set that we compute is destined for
    /// `hwloc_obj_add_other_obj_sets()` consumption, we do not actually need to
    /// do anything to expand the group so that it follows the consistency
    /// rules. What requires work on our side is rejecting inconsistent groups
    /// and adding objects to represent the group hwloc would actually create.
    ///
    /// Consequently, there is a `make_hwloc_input` operating mode which only
    /// checks groups for consistency and does not perform group expansion,
    /// meant, for situations where we do not care about group members but only
    /// about the union of their cpusets/nodesets.
    ///
    /// # Errors
    ///
    /// - [`Inconsistent`] if [`GroupChildFilter::Mixed`] was used in strict
    ///   mode, but the selected normal and memory object sets were not
    ///   consistent.
    ///
    /// [`Inconsistent`]: InsertGroupError::Inconsistent
    pub(self) fn filter_children<'topology>(
        &mut self,
        parent: &'topology TopologyObject,
        make_hwloc_input: bool,
    ) -> Result<Vec<&'topology TopologyObject>, InsertGroupError> {
        /// Shorthand to get to the cpuset of a normal or memory child
        fn child_cpuset(child: &TopologyObject) -> BitmapRef<'_, CpuSet> {
            child
                .cpuset()
                .expect("normal & memory children should have cpusets")
        }
        /// Shorthand to get to the nodeset of a normal or memory child
        fn child_nodeset(child: &TopologyObject) -> BitmapRef<'_, NodeSet> {
            child
                .nodeset()
                .expect("normal & memory children should have nodesets")
        }

        // Pick user-requested group members, only check for group consistency
        // in strict mode
        let mut children = Vec::new();
        match self {
            Self::Normal(filter) => {
                children.extend(parent.normal_children().filter(|obj| filter(obj)));
            }
            Self::Memory(filter) => {
                children.extend(parent.memory_children().filter(|obj| filter(obj)));
            }
            Self::Mixed {
                strict,
                normal,
                memory,
            } => {
                children.extend(parent.normal_children().filter(|obj| normal(obj)));
                if *strict {
                    // In strict mixed mode, we need to check that hwloc won't
                    // add extra objects the users didn't expect to the group
                    let normal_cpuset = children.iter().fold(CpuSet::new(), |mut acc, child| {
                        acc |= child_cpuset(child);
                        acc
                    });
                    for memory_child in parent.memory_children() {
                        let memory_cpuset = child_cpuset(memory_child);
                        if memory(memory_child) {
                            // If a memory child is picked, then hwloc will add
                            // all of its CPU children to the group, so the user
                            // should have added them to the normal child set.
                            if !normal_cpuset.includes(memory_cpuset) {
                                return Err(InsertGroupError::Inconsistent);
                            }
                            children.push(memory_child);
                        } else {
                            // If a memory child has CPU children and all of
                            // them are picked, then hwloc will add the memory
                            // object to the group, so the user should have
                            // added it to the memory child set.
                            if !memory_cpuset.is_empty() && normal_cpuset.includes(memory_cpuset) {
                                return Err(InsertGroupError::Inconsistent);
                            }
                        }
                    }
                } else {
                    children.extend(parent.memory_children().filter(|obj| memory(obj)));
                }
            }
        }

        // If the output is user-visible, as opposed to being just for hwloc
        // consumption, make child set match the group hwloc would create.
        if !make_hwloc_input {
            let (group_cpuset, group_nodeset) = children.drain(..).fold(
                (CpuSet::new(), NodeSet::new()),
                |(mut cpuset, mut nodeset), child| {
                    cpuset |= child_cpuset(child);
                    nodeset |= child_nodeset(child);
                    (cpuset, nodeset)
                },
            );
            children.extend(
                parent
                    .normal_children()
                    .chain(parent.memory_children())
                    .filter(|child| {
                        group_cpuset.includes(child_cpuset(child))
                            && group_nodeset.includes(child_nodeset(child))
                    }),
            );
        }
        Ok(children)
    }
}
//
impl<NormalFilter, MemoryFilter> Debug for GroupChildFilter<NormalFilter, MemoryFilter>
where
    NormalFilter: FnMut(&TopologyObject) -> bool,
    MemoryFilter: FnMut(&TopologyObject) -> bool,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Normal(_) => f.debug_struct("Normal").finish_non_exhaustive(),
            Self::Memory(_) => f.debug_struct("Memory").finish_non_exhaustive(),
            Self::Mixed { strict, .. } => f
                .debug_struct("Mixed")
                .field("strict", &strict)
                .finish_non_exhaustive(),
        }
    }
}

/// Error while creating a [`Group`](ObjectType::Group) object
#[derive(Clone, Debug, Eq, Error, Hash, PartialEq)]
pub enum InsertGroupError {
    /// Attempted to create a group in a topology where groups are filtered out
    ///
    /// This happens when the type filter for [`ObjectType::Group`] is set to
    /// [`TypeFilter::KeepNone`].
    #[error("can't create Group objects when their type filter is KeepNone")]
    FilteredOut,

    /// Specified parent is not a normal object
    ///
    /// Group objects are normal objects, and a normal object may only have
    /// another normal object as a parent, therefore the designated parent of a
    /// group has to be a normal object.
    #[error("group object parent has non-normal object type {0}")]
    BadParentType(ObjectType),

    /// Specified parent does not belong to this topology
    ///
    /// It is not okay to take an object from a different topology when asked to
    /// specify a group's parent.
    #[error("group object parent {0}")]
    ForeignParent(#[from] ForeignObjectError),

    /// Attempted to create a group without children
    ///
    /// The position of group objects in the topology is defined by their child
    /// set, therefore a group object cannot be empty.
    #[error("a group must have at least one child object")]
    Empty,

    /// Attempted to create an inconsistent group by using
    /// [`GroupChildFilter::Mixed`] in strict mode
    ///
    /// The group child set you asked for cannot be handled in hwloc's current
    /// group creation model, without adding extra objects to the group.
    #[error("attempted to create an inconsistent group (see GroupChildFilter docs)")]
    Inconsistent,
}

/// RAII guard for `Group` objects that have been allocated, but not inserted
///
/// Ensures that these groups are auto-deleted if not inserted for any reason
/// (typically as a result of erroring out).
///
/// # Safety
///
/// `group` must be a newly allocated, not-yet-inserted `Group` object that is
/// bound to topology editor `editor`. It would be an `&mut TopologyObject` if
/// this didn't break the Rust aliasing rules.
struct AllocatedGroup<'editor, 'topology> {
    /// Group object
    group: NonNull<TopologyObject>,

    /// Underlying [`TopologyEditor`] the Group is allocated from
    editor: &'editor mut TopologyEditor<'topology>,
}
//
impl<'editor, 'topology> AllocatedGroup<'editor, 'topology> {
    /// Allocate a new Group object
    pub(self) fn new(
        editor: &'editor mut TopologyEditor<'topology>,
    ) -> Result<Self, RawHwlocError> {
        // SAFETY: - Topology is trusted to contain a valid ptr (type invariant)
        //         - hwloc ops are trusted to keep *mut parameters in a
        //           valid state unless stated otherwise
        errors::call_hwloc_ptr_mut("hwloc_topology_alloc_group_object", || unsafe {
            hwlocality_sys::hwloc_topology_alloc_group_object(editor.topology_mut_ptr())
        })
        .map(|group| Self {
            // SAFETY: - hwloc is trusted to produce a valid, non-inserted group
            //           object pointer
            //         - AsNewtype is trusted to be implemented correctly
            group: unsafe { group.as_newtype() },
            editor,
        })
    }

    /// Expand cpu sets and node sets to cover designated children
    ///
    /// This is only meant to be executed once. The children consistency checks
    /// assume the input child set is the full child set and adding children
    /// below multiple parents is not supported.
    ///
    /// # Errors
    ///
    /// - [`BadParentType`] if the designated group parent is not a normal
    ///   object.
    /// - [`ForeignParent`] if the designated group parent does not belong to
    ///   the topology that is being edited.
    /// - [`Empty`] if the [`GroupChildFilter`] did not select any child.
    /// - [`Inconsistent`] if [`GroupChildFilter::Mixed`] was used in strict
    ///   mode, but the selected normal and memory object sets were not
    ///   consistent.
    ///
    /// [`BadParentType`]: InsertGroupError::BadParentType
    /// [`Empty`]: InsertGroupError::Empty
    /// [`ForeignParent`]: InsertGroupError::ForeignParent
    /// [`Inconsistent`]: InsertGroupError::Inconsistent
    pub(self) fn add_children<NormalFilter, MemoryFilter>(
        &mut self,
        find_parent: impl FnOnce(&Topology) -> &TopologyObject,
        mut child_filter: GroupChildFilter<NormalFilter, MemoryFilter>,
    ) -> Result<(), InsertGroupError>
    where
        NormalFilter: FnMut(&TopologyObject) -> bool,
        MemoryFilter: FnMut(&TopologyObject) -> bool,
    {
        // Pick the group's parent
        let topology = self.editor.topology();
        let parent = find_parent(topology);
        if !parent.object_type().is_normal() {
            return Err(InsertGroupError::BadParentType(parent.object_type()));
        }
        if !topology.contains(parent) {
            return Err(InsertGroupError::ForeignParent(parent.into()));
        }

        // Enumerate children
        let children = child_filter.filter_children(parent, true)?;
        if children.is_empty() {
            return Err(InsertGroupError::Empty);
        }

        /// Polymorphized subset of this function (avoids generics code bloat)
        ///
        /// # Safety
        ///
        /// - `group` must point to the inner group of an [`AllocatedGroup`]
        /// - `children` must have been checked to belong to the topology of
        ///   said [`AllocatedGroup`]
        unsafe fn polymorphized(group: NonNull<TopologyObject>, children: Vec<&TopologyObject>) {
            // Add children to this group
            for child in children {
                let result =
                    // SAFETY: - group is assumed to be valid as a type
                    //           invariant of AllocatedGroup
                    //         - hwloc ops are trusted not to modify *const
                    //           parameters
                    //         - child was checked to belong to the same
                    //           topology as group
                    //         - AsInner is trusted to be implemented correctly
                    errors::call_hwloc_int_normal("hwloc_obj_add_other_obj_sets", || unsafe {
                        hwlocality_sys::hwloc_obj_add_other_obj_sets(
                            group.as_inner().as_ptr(),
                            child.as_inner(),
                        )
                    });
                let handle_enomem =
                    |raw_err: RawHwlocError| panic!("Internal reallocation failed: {raw_err}");
                match result {
                    Ok(_) => {}
                    Err(
                        raw_err @ RawHwlocError {
                            errno: Some(errno::Errno(ENOMEM)),
                            ..
                        },
                    ) => handle_enomem(raw_err),
                    #[cfg(windows)]
                    Err(raw_err @ RawHwlocError { errno: None, .. }) => {
                        // As explained in the RawHwlocError documentation,
                        // errno values may not correctly propagate from hwloc
                        // to hwlocality on Windows. Since there is only one
                        // expected errno value here, we'll interpret lack of
                        // errno as ENOMEM on Windows.
                        handle_enomem(raw_err)
                    }
                    Err(raw_err) => unreachable!("Unexpected hwloc error: {raw_err}"),
                }
            }
        }

        // Call into the polymorphized function
        // SAFETY: - This is indeed the inner group of this AllocatedGroup
        //         - children can only belong to this topology
        unsafe { polymorphized(self.group, children) };
        Ok(())
    }

    /// Configure hwloc's group merging policy
    ///
    /// By default, hwloc may or may not merge identical groups covering the
    /// same objects. You can encourage or inhibit this tendency with this method.
    pub(self) fn configure_merging(&mut self, dont_merge: bool) {
        let group_attributes: &mut GroupAttributes =
            // SAFETY: - We know this is a group object as a type invariant, so
            //           accessing the group raw attribute is safe
            //         - We trust hwloc to have initialized the group attributes
            //           to a valid state
            //         - We are not changing the raw attributes variant
            unsafe { (&mut (*self.group.as_mut().as_inner().attr).group).as_newtype() };
        if dont_merge {
            // Make sure the new group is not merged with an existing object
            group_attributes.prevent_merging();
        } else {
            // Make sure the new group is deterministically always merged with
            // existing groups that have the same locality.
            group_attributes.favor_merging();
        }
    }

    /// Insert this Group object into the underlying topology
    ///
    /// # Errors
    ///
    /// Will return an unspecified error if any of the following happens:
    ///
    /// - Insertion failed because of conflicting sets in the topology tree
    /// - Group objects are filtered out of the topology via
    ///   [`TypeFilter::KeepNone`]
    /// - The object was discarded because no set was initialized in the Group,
    ///   or they were all empty.
    pub(self) fn insert(mut self) -> Result<InsertedGroup<'topology>, RawHwlocError> {
        // SAFETY: self is forgotten after this, so no drop or reuse will occur
        let res = unsafe { self.insert_impl() };
        std::mem::forget(self);
        res
    }

    /// Implementation of `insert()` with an `&mut self` argument
    ///
    /// # Errors
    ///
    /// Will return an unspecified error if any of the following happens:
    ///
    /// - Insertion failed because of conflicting sets in the topology tree
    /// - Group objects are filtered out of the topology via
    ///   [`TypeFilter::KeepNone`]
    /// - The object was discarded because no set was initialized in the Group,
    ///   or they were all empty.
    ///
    /// # Safety
    ///
    /// After calling this method, `self` is in an invalid state and should not
    /// be used in any way anymore. In particular, care should be taken to
    /// ensure that its Drop destructor is not called.
    unsafe fn insert_impl(&mut self) -> Result<InsertedGroup<'topology>, RawHwlocError> {
        // SAFETY: - Topology is trusted to contain a valid ptr (type invariant)
        //         - Inner group pointer is assumed valid as a type invariant
        //         - hwloc ops are trusted not to modify *const parameters
        //         - hwloc ops are trusted to keep *mut parameters in a
        //           valid state unless stated otherwise
        //         - We break the AllocatedGroup type invariant by inserting the
        //           group object, but a precondition warns the user about it
        //         - AsInner is trusted to be implemented correctly
        errors::call_hwloc_ptr_mut("hwloc_topology_insert_group_object", || unsafe {
            hwlocality_sys::hwloc_topology_insert_group_object(
                self.editor.topology_mut_ptr(),
                self.group.as_inner().as_ptr(),
            )
        })
        .map(|mut result| {
            if result == self.group.as_inner() {
                // SAFETY: - We know this is a group object as a type invariant
                //         - Output lifetime is bound to the topology it comes from
                //         - Group has been successfully inserted, can expose &mut
                InsertedGroup::New(unsafe { self.group.as_mut() })
            } else {
                // SAFETY: - Successful result is trusted to point to an
                //           existing group, in a valid state
                //         - Output lifetime is bound to the topology it comes from
                InsertedGroup::Existing(unsafe { result.as_mut().as_newtype() })
            }
        })
    }
}
//
impl Drop for AllocatedGroup<'_, '_> {
    #[allow(clippy::print_stderr)]
    fn drop(&mut self) {
        // Since hwloc v2.10 there is a way to cleanly free group objects
        #[cfg(feature = "hwloc-2_10_0")]
        {
            let result = errors::call_hwloc_int_normal(
                "hwloc_topology_free_group_object",
                // SAFETY: - Inner group pointer is assumed valid as a type invariant
                //         - The state where this invariant is broken, produced
                //           by Self::insert_impl() or
                //           hwloc_topology_free_group_object(), is never
                //           exposed to Drop.
                //         - This invalidates the AllocatedGroup, but that's
                //           fine since it is not reachable after Drop
                || unsafe {
                    hwlocality_sys::hwloc_topology_free_group_object(
                        self.editor.topology_mut_ptr(),
                        self.group.as_inner().as_ptr(),
                    )
                },
            );
            if let Err(e) = result {
                eprintln!("ERROR: Failed to deallocate group object ({e}).");
            }
        }

        // Before hwloc v2.10, there was no API to delete a previously allocated
        // group object without attempting to insert it into the topology in a
        // configuration with empty sets, which is guaranteed to fail.
        #[cfg(not(feature = "hwloc-2_10_0"))]
        {
            // SAFETY: - Inner group pointer is assumed valid as a type invariant
            //         - The state where this invariant is invalidated, produced by
            //           insert_impl(), is never exposed to Drop
            unsafe {
                TopologyObject::delete_all_sets(self.group);
            }
            // SAFETY: This invalidates the AllocatedGroup, but that's fine
            //         since it is not reachable after Drop
            if unsafe { self.insert_impl().is_ok() } {
                eprintln!("ERROR: Failed to deallocate group object.");
            }
        }
    }
}

/// Result of inserting a Group object
#[derive(Debug)]
#[must_use]
pub enum InsertedGroup<'topology> {
    /// New Group that was properly inserted
    New(&'topology mut TopologyObject),

    /// Existing object that already fulfilled the role of the proposed Group
    Existing(&'topology TopologyObject),
}

/// Error returned by [`TopologyEditor::insert_misc_object()`]
#[derive(Clone, Debug, Eq, Error, Hash, PartialEq)]
pub enum InsertMiscError {
    /// Attempted to create a Misc object in a topology where they are filtered
    /// out
    ///
    /// This happens when the type filter for [`ObjectType::Misc`] is set to
    /// [`TypeFilter::KeepNone`].
    #[error("can't create Misc objects when their type filter is KeepNone")]
    FilteredOut,

    /// Specified parent does not belong to this topology
    #[error("Misc object parent {0}")]
    ForeignParent(#[from] ForeignObjectError),

    /// Object name contains NUL chars, which hwloc can't handle
    #[error("Misc object name can't contain NUL chars")]
    NameContainsNul,

    /// Object name is already present in the topology
    #[error("Requested Misc object name already exists in the topology")]
    NameAlreadyExists,
}
//
impl From<NulError> for InsertMiscError {
    fn from(_: NulError) -> Self {
        Self::NameContainsNul
    }
}

// NOTE: Do not implement traits like AsRef/Deref/Borrow for TopologyEditor,
//       that would be unsafe as it would expose &Topology with unevaluated lazy
//       hwloc caches, and calling their methods could violates Rust's aliasing
//       model via mutation through &Topology.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        object::{
            depth::{Depth, NormalDepth},
            TopologyObjectID,
        },
        strategies::{any_object, any_string, topology_related_set},
    };
    use proptest::prelude::*;
    use similar_asserts::assert_eq;
    use std::{
        collections::{BTreeMap, HashMap, HashSet},
        ffi::CStr,
        fmt::Debug,
        panic::RefUnwindSafe,
        sync::OnceLock,
    };

    /// Make sure opening/closing the editor doesn't affect the topology
    #[test]
    fn basic_lifecycle() {
        let reference = Topology::test_instance();
        let mut topology = reference.clone();
        topology.edit(|editor| {
            assert_eq!(editor.topology(), reference);
        });
        assert_eq!(&topology, reference);
    }

    // --- Test topology restrictions ---

    proptest! {
        #[test]
        fn restrict_cpuset(
            cpuset in topology_related_set(Topology::cpuset),
            flags: RestrictFlags,
        ) {
            check_restrict(Topology::test_instance(), &cpuset, flags)?;
        }

        #[test]
        fn restrict_nodeset(
            nodeset in topology_related_set(Topology::nodeset),
            flags: RestrictFlags,
        ) {
            check_restrict(Topology::test_instance(), &nodeset, flags)?;
        }
    }

    /// Set-generic test for [`TopologyEditor::restrict()`]
    fn check_restrict<Set: SpecializedBitmap + RefUnwindSafe>(
        initial_topology: &Topology,
        restrict_set: &Set,
        flags: RestrictFlags,
    ) -> Result<(), TestCaseError> {
        // Compute the restricted topology
        let mut final_topology = initial_topology.clone();
        let result = final_topology.edit(|editor| editor.restrict(restrict_set, flags));

        // Abstract over the kind of set that is being restricted
        let topology_sets = |topology| ErasedSets::from_topology::<Set>(topology);
        let object_sets = |obj: &TopologyObject| ErasedSets::from_object::<Set>(obj);
        let predict_final_sets = |initial_sets: &ErasedSets| {
            initial_sets.predict_restricted(initial_topology, restrict_set)
        };

        // Predict the effect of topology restriction
        let initial_sets = topology_sets(initial_topology);
        let predicted_sets = predict_final_sets(&initial_sets);

        // If one attempts to remove all CPUs and NUMA nodes, and error will be
        // returned and the topology will be unchanged
        if predicted_sets.target.is_empty() {
            prop_assert_eq!(result, Err(ParameterError::from(restrict_set.clone())));
            prop_assert_eq!(initial_topology, &final_topology);
            return Ok(());
        }
        result.unwrap();

        // Otherwise, the topology sets should be restricted as directed
        let final_sets = topology_sets(&final_topology);
        prop_assert_eq!(&final_sets, &predicted_sets);

        // Removing no CPU or node leaves the topology unchanged
        if final_sets == initial_sets {
            prop_assert_eq!(initial_topology, &final_topology);
            return Ok(());
        }

        // Now we're going to predict the outcome on topology objects
        let parent_id =
            |obj: &TopologyObject| obj.parent().map(TopologyObject::global_persistent_index);
        let predict_object =
            |obj: &TopologyObject, predicted_parent_id: Option<TopologyObjectID>| {
                PredictedObject::new(
                    obj,
                    predicted_parent_id,
                    object_sets(obj).map(|sets| predict_final_sets(&sets)),
                )
            };
        let mut predicted_objects = BTreeMap::new();

        // First predict the set of normal and memory objects. Start by
        // including or excluding leaf PU and NUMA node objects...
        let id = |obj: &TopologyObject| obj.global_persistent_index();
        let mut retained_leaves = initial_topology
            .objects_with_type(ObjectType::PU)
            .chain(initial_topology.objects_at_depth(Depth::NUMANode))
            .filter(|obj| {
                let predicted_sets = predict_final_sets(&object_sets(obj).unwrap());
                !(predicted_sets.target.is_empty()
                    && (predicted_sets.other.is_empty()
                        || flags.contains(RestrictFlags::REMOVE_EMPTIED)))
            })
            .map(|obj| (id(obj), obj))
            .collect::<HashMap<_, _>>();

        // ...then recurse into parents to cover the object tree
        let mut next_leaves = HashMap::new();
        while !retained_leaves.is_empty() {
            for (obj_id, obj) in retained_leaves.drain() {
                predicted_objects.insert(obj_id, predict_object(obj, parent_id(obj)));
                if let Some(parent) = obj.parent() {
                    next_leaves.insert(id(parent), parent);
                }
            }
            std::mem::swap(&mut retained_leaves, &mut next_leaves);
        }

        // When their normal parent is destroyed, I/O and Misc objects may
        // either, depending on flags, be deleted or re-attached to the
        // lowest-depth ancestor object that is still present in the topology.
        let rebind_parent = |obj: &TopologyObject| {
            let mut parent = obj.parent().unwrap();
            if !(parent.object_type().is_io() || predicted_objects.contains_key(&id(parent))) {
                parent = parent
                    .ancestors()
                    .find(|ancestor| predicted_objects.contains_key(&id(ancestor)))
                    .unwrap()
            }
            Some(id(parent))
        };

        // Predict the fate I/O objects, including deletions and rebinding
        let io_objects = initial_topology
            .io_objects()
            .filter(|obj| {
                if flags.contains(RestrictFlags::ADAPT_IO) {
                    obj.ancestors()
                        .any(|ancestor| predicted_objects.contains_key(&id(ancestor)))
                } else {
                    predicted_objects.contains_key(&id(obj.first_non_io_ancestor().unwrap()))
                }
            })
            .map(|obj| (id(obj), predict_object(obj, rebind_parent(obj))))
            .collect::<Vec<_>>();

        // Predict the fate of Misc objects using a similar logic
        let misc_objects = initial_topology
            .objects_with_type(ObjectType::Misc)
            .filter(|obj| {
                flags.contains(RestrictFlags::ADAPT_MISC) || {
                    predicted_objects.contains_key(&id(obj.parent().unwrap()))
                }
            })
            .map(|obj| (id(obj), predict_object(obj, rebind_parent(obj))))
            .collect::<Vec<_>>();
        predicted_objects.extend(io_objects);
        predicted_objects.extend(misc_objects);

        // Finally, check that the final object set matches our prediction
        let final_objects = final_topology
            .objects()
            .map(|obj| {
                (
                    id(obj),
                    PredictedObject::new(obj, parent_id(obj), object_sets(obj)),
                )
            })
            .collect::<BTreeMap<_, _>>();
        prop_assert_eq!(predicted_objects, final_objects);
        Ok(())
    }

    /// [`CpuSet`]/[`NodeSet`] abstraction layer
    #[derive(Clone, Debug, Eq, PartialEq)]
    struct ErasedSets {
        /// Set that is being restricted
        target: Bitmap,

        /// Set that is indirectly affected by the restriction
        other: Bitmap,
    }
    //
    impl ErasedSets {
        /// Get [`ErasedSets`] from a [`Topology`]
        fn from_topology<RestrictedSet: SpecializedBitmap>(topology: &Topology) -> Self {
            match RestrictedSet::BITMAP_KIND {
                BitmapKind::CpuSet => Self {
                    target: Self::ref_to_bitmap(topology.cpuset()),
                    other: Self::ref_to_bitmap(topology.nodeset()),
                },
                BitmapKind::NodeSet => Self {
                    target: Self::ref_to_bitmap(topology.nodeset()),
                    other: Self::ref_to_bitmap(topology.cpuset()),
                },
            }
        }

        /// Get [`ErasedSets`] from a [`TopologyObject`]
        fn from_object<RestrictedSet: SpecializedBitmap>(obj: &TopologyObject) -> Option<Self> {
            Some(match RestrictedSet::BITMAP_KIND {
                BitmapKind::CpuSet => Self {
                    target: Self::ref_to_bitmap(obj.cpuset()?),
                    other: Self::ref_to_bitmap(obj.nodeset().unwrap()),
                },
                BitmapKind::NodeSet => Self {
                    target: Self::ref_to_bitmap(obj.nodeset()?),
                    other: Self::ref_to_bitmap(obj.cpuset().unwrap()),
                },
            })
        }

        /// Predict the [`ErasedSets`] after restricting the source topology
        fn predict_restricted<RestrictedSet: SpecializedBitmap>(
            &self,
            initial_topology: &Topology,
            restrict_set: &RestrictedSet,
        ) -> Self {
            let restrict_set: Bitmap = restrict_set.clone().into();
            let predicted_target = &self.target & restrict_set;
            let predicted_other = match RestrictedSet::BITMAP_KIND {
                BitmapKind::CpuSet => {
                    let predicted_target = CpuSet::from(predicted_target.clone());
                    Bitmap::from(NodeSet::from_cpuset(initial_topology, &predicted_target))
                }
                BitmapKind::NodeSet => {
                    let predicted_target = NodeSet::from(predicted_target.clone());
                    Bitmap::from(CpuSet::from_nodeset(initial_topology, &predicted_target))
                }
            };
            Self {
                target: predicted_target,
                other: predicted_other,
            }
        }

        /// Convert a [`BitmapRef`] to a type-erased [`Bitmap`]
        fn ref_to_bitmap<Set: SpecializedBitmap>(set: BitmapRef<'_, Set>) -> Bitmap {
            set.clone_target().into()
        }
    }

    /// Predicted topology object properties after topology restriction
    #[derive(Clone, Debug, Eq, PartialEq)]
    struct PredictedObject {
        object_type: ObjectType,
        subtype: Option<String>,
        name: Option<String>,
        attributes: Option<String>,
        os_index: Option<usize>,
        depth: Depth,
        parent_id: Option<TopologyObjectID>,
        sets: Option<ErasedSets>,
        infos: String,
    }
    //
    impl PredictedObject {
        /// Given some predicted properties, predict the rest
        fn new(
            obj: &TopologyObject,
            parent_id: Option<TopologyObjectID>,
            sets: Option<ErasedSets>,
        ) -> Self {
            let stringify = |s: Option<&CStr>| s.map(|s| s.to_string_lossy().to_string());
            Self {
                object_type: obj.object_type(),
                subtype: stringify(obj.subtype()),
                name: stringify(obj.name()),
                attributes: obj.attributes().map(|attr| format!("{attr:?}")),
                os_index: obj.os_index(),
                depth: obj.depth(),
                parent_id,
                sets,
                infos: format!("{:?}", obj.infos().iter().collect::<Vec<_>>()),
            }
        }
    }

    // --- Changing the set of allowed PUs and NUMA nodes ---

    proptest! {
        /// Test AllowSet construction from CpuSet
        #[test]
        fn allowset_from_cpuset(cpuset: CpuSet) {
            let allow_set = AllowSet::from(&cpuset);
            let AllowSet::Custom { cpuset: Some(allow_cpuset), nodeset: None } = allow_set else {
                panic!("Unexpected allow set {allow_set}");
            };
            prop_assert_eq!(allow_cpuset, &cpuset);
        }

        /// Test AllowSet construction from NodeSet
        #[test]
        fn allowset_from_nodeset(nodeset: NodeSet) {
            let allow_set = AllowSet::from(&nodeset);
            let AllowSet::Custom { cpuset: None, nodeset: Some(allow_nodeset) } = allow_set else {
                panic!("Unexpected allow set {allow_set}");
            };
            prop_assert_eq!(allow_nodeset, &nodeset);
        }
    }

    /// Owned version of [`AllowSet`]
    #[derive(Clone, Debug, Eq, Hash, PartialEq)]
    enum OwnedAllowSet {
        All,
        LocalRestrictions,
        Custom {
            cpuset: Option<CpuSet>,
            nodeset: Option<NodeSet>,
        },
    }
    //
    impl OwnedAllowSet {
        /// Borrow an [`AllowSet`] from this
        fn as_allow_set(&self) -> AllowSet<'_> {
            match self {
                Self::All => AllowSet::All,
                Self::LocalRestrictions => AllowSet::LocalRestrictions,
                Self::Custom { cpuset, nodeset } => AllowSet::Custom {
                    cpuset: cpuset.as_ref(),
                    nodeset: nodeset.as_ref(),
                },
            }
        }
    }

    /// Generate an `OwnedAllowSet` for `TopologyEditor::allow()` testing
    fn any_allow_set() -> impl Strategy<Value = OwnedAllowSet> {
        fn topology_related_set_opt<Set: SpecializedBitmap>(
            topology_set: impl FnOnce(&Topology) -> BitmapRef<'_, Set>,
        ) -> impl Strategy<Value = Option<Set>> {
            prop_oneof![
                3 => topology_related_set(topology_set).prop_map(Some),
                2 => Just(None)
            ]
        }
        prop_oneof![
            1 => Just(OwnedAllowSet::All),
            1 => Just(OwnedAllowSet::LocalRestrictions),
            3 => (
                topology_related_set_opt(Topology::complete_cpuset),
                topology_related_set_opt(Topology::complete_nodeset)
            ).prop_map(|(cpuset, nodeset)| OwnedAllowSet::Custom {
                cpuset, nodeset
            })
        ]
    }

    proptest! {
        /// Test display implementation of AllowSet
        #[test]
        fn allowset_display(owned_allow_set in any_allow_set()) {
            let allow_set = owned_allow_set.as_allow_set();
            let display = allow_set.to_string();
            match allow_set {
                AllowSet::All => prop_assert_eq!(display, "All"),
                AllowSet::LocalRestrictions => prop_assert_eq!(display, "LocalRestrictions"),
                AllowSet::Custom { cpuset: Some(cset), nodeset: Some(nset) } => {
                    prop_assert_eq!(display, format!("Custom({cset}, {nset})"))
                }
                AllowSet::Custom { cpuset: Some(cset), nodeset: None } => {
                    prop_assert_eq!(display, format!("Custom({cset})"))
                }
                AllowSet::Custom { cpuset: None, nodeset: Some(nset) } => {
                    prop_assert_eq!(display, format!("Custom({nset})"))
                }
                AllowSet::Custom { cpuset: None, nodeset: None } => {
                    prop_assert_eq!(display, "Custom()")
                }
            }
        }

        /// Test [`TopologyEditor::allow()`]
        #[test]
        fn allow(owned_allow_set in any_allow_set()) {
            let initial_topology = Topology::test_instance();
            let mut topology = initial_topology.clone();

            let allow_set = owned_allow_set.as_allow_set();
            let result = topology.edit(|editor| editor.allow(allow_set));

            // Only a couple OSes support AllowSet::LocalRestrictions
            const OS_SUPPORTS_LOCAL_RESTRICTIONS: bool = cfg!(any(target_os = "linux", target_os = "solaris"));

            match allow_set {
                AllowSet::All => {
                    result.unwrap();
                    prop_assert_eq!(topology.allowed_cpuset(), topology.cpuset());
                    prop_assert_eq!(topology.allowed_nodeset(), topology.nodeset());
                }
                AllowSet::LocalRestrictions => {
                    // LocalRestrictions is only supported on Linux
                    if !OS_SUPPORTS_LOCAL_RESTRICTIONS {
                        match result {
                            Err(HybridError::Rust(AllowSetError::Unsupported)) => {}
                            #[cfg(windows)]
                            Err(HybridError::Hwloc(RawHwlocError { errno: None, .. })) => {}
                            other => panic!("unexpected result {other:?}"),
                        }
                        return Ok(());
                    }

                    // LocalRestrictions does what the normal
                    // topology-building process does, so it has no observable
                    // effect on a freshly built topology, but see below.
                    result.unwrap();
                    prop_assert_eq!(&topology, initial_topology);
                }
                AllowSet::Custom { cpuset, nodeset } => {
                    if cpuset.is_none() && nodeset.is_none() {
                        prop_assert_eq!(result, Err(AllowSetError::EmptyCustom.into()));
                        return Ok(());
                    }

                    let mut effective_cpuset = topology.cpuset().clone_target();
                    if let Some(cpuset) = cpuset {
                        effective_cpuset &= cpuset;
                        if effective_cpuset.is_empty() {
                            prop_assert_eq!(result, Err(AllowSetError::InvalidCpuset.into()));
                            return Ok(());
                        }
                    }

                    let mut effective_nodeset = topology.nodeset().clone_target();
                    if let Some(nodeset) = nodeset {
                        effective_nodeset &= nodeset;
                        if effective_nodeset.is_empty() {
                            prop_assert_eq!(result, Err(AllowSetError::InvalidNodeset.into()));
                            return Ok(());
                        }
                    }

                    result.unwrap();
                    prop_assert_eq!(topology.allowed_cpuset(), effective_cpuset);
                    prop_assert_eq!(topology.allowed_nodeset(), effective_nodeset);
                }
            }

            // Here we check that LocalRestrictions resets the topology from any
            // allow set we may have configured back to its original allow sets.
            if OS_SUPPORTS_LOCAL_RESTRICTIONS {
                let result = topology.edit(|editor| editor.allow(AllowSet::LocalRestrictions));
                result.unwrap();
                prop_assert_eq!(&topology, initial_topology);
            }
        }
    }

    // --- Grouping objects ---

    /// Check [`GroupChildFilter`]'s debug printout
    #[test]
    fn child_filter_debug() {
        let filter = |_: &TopologyObject| true;
        assert_eq!(
            format!("{:?}", GroupChildFilter::normal(filter)),
            "Normal { .. }"
        );
        assert_eq!(
            format!("{:?}", GroupChildFilter::memory(filter)),
            "Memory { .. }"
        );
        assert_eq!(
            format!(
                "{:?}",
                GroupChildFilter::Mixed {
                    strict: false,
                    normal: filter,
                    memory: filter
                }
            ),
            "Mixed { strict: false, .. }"
        );
        assert_eq!(
            format!(
                "{:?}",
                GroupChildFilter::Mixed {
                    strict: true,
                    normal: filter,
                    memory: filter
                }
            ),
            "Mixed { strict: true, .. }"
        );
    }

    /// Child filtering function, as a trait object
    type DynChildFilter = Box<dyn FnMut(&TopologyObject) -> bool + UnwindSafe>;

    /// Within the test topology, pick a parent, a set of group members, and a
    /// group merging configuration
    fn group_building_blocks() -> impl Strategy<
        Value = (
            &'static TopologyObject,
            GroupChildFilter<DynChildFilter, DynChildFilter>,
            bool,
        ),
    > {
        // Pick a parent for the group object
        let any_parent = prop_oneof! [
            3 => multi_child_parent(),
            2 => any_object(),
        ];

        // Given a parent, pick a child set
        any_parent.prop_flat_map(move |parent| {
            let child_filter = child_filter_from_parent(parent);
            (child_filter, any::<bool>())
                .prop_map(move |(child_filter, merge)| (parent, child_filter, merge))
        })
    }

    /// Pick a parent for which group object creation can succeed
    ///
    /// The `find_parent` callback to `insert_group_object` could return any
    /// object as a parent, including objects from different topologies. But
    /// outside of the `dont_merge` special case, group creation will fail or
    /// return `Existing` if the parent object is anything but a normal object
    /// with >= 2 children. This function only picks parents which match this
    /// criterion, and is used to bias the RNG towards more successful group
    /// generation.
    ///
    /// Furthermore, parents at high depths like CPU cores are more numerous
    /// than objects at low depths like L3 cache. Therefore, a random pick
    /// with a uniform distribution is a lot more likely to pick high-depth
    /// parents than low-depth parents. To give low-depth parents a fair amount
    /// of test coverage, we bias the parent distribution such that each parent
    /// depth has an equal chance of coming up.
    fn multi_child_parent() -> impl Strategy<Value = &'static TopologyObject> {
        let topology = Topology::test_instance();

        let good_parents_by_depth = NormalDepth::iter_range(NormalDepth::MIN, topology.depth())
            .filter_map(|depth| {
                let good_parents = topology
                    .objects_at_depth(depth)
                    .filter(|obj| obj.normal_arity() >= 2 || obj.memory_arity() >= 2)
                    .collect::<Vec<_>>();
                (!good_parents.is_empty()).then_some((depth, good_parents))
            })
            .collect::<HashMap<_, _>>();

        let good_parent_depths = good_parents_by_depth.keys().copied().collect::<Vec<_>>();
        prop::sample::select(good_parent_depths)
            .prop_flat_map(move |depth| prop::sample::select(good_parents_by_depth[&depth].clone()))
    }

    /// Given a group parent, generate child filters
    fn child_filter_from_parent(
        parent: &TopologyObject,
    ) -> impl Strategy<Value = GroupChildFilter<DynChildFilter, DynChildFilter>> {
        // Turn normal and memory child list of parent into 'static objects
        // using their global persistent ID
        fn children_ids<'a>(
            children: impl Iterator<Item = &'a TopologyObject>,
        ) -> Vec<TopologyObjectID> {
            children
                .map(TopologyObject::global_persistent_index)
                .collect::<Vec<_>>()
        }
        let normal_ids = children_ids(parent.normal_children());
        let memory_ids = children_ids(parent.memory_children());

        // Normal and memory child filtering configurations
        let normal_child_subset = || child_subset(normal_ids.clone());
        let normal_child_filter =
            || normal_child_subset().prop_map(|subset| GroupChildFilter::Normal(filter_fn(subset)));
        let memory_child_subset = || child_subset(memory_ids.clone());
        let memory_child_filter =
            || memory_child_subset().prop_map(|subset| GroupChildFilter::Memory(filter_fn(subset)));

        // Final child filter generation strategy
        prop_oneof![
            normal_child_filter(),
            memory_child_filter(),
            (any::<bool>(), normal_child_subset(), memory_child_subset()).prop_map(
                |(strict, normal_subset, memory_subset)| {
                    GroupChildFilter::Mixed {
                        strict,
                        normal: filter_fn(normal_subset),
                        memory: filter_fn(memory_subset),
                    }
                }
            )
        ]
    }

    /// Given one of the parent `TopologyObject`'s children lists, select a
    /// subset of it
    ///
    /// There is a bias towards picking no children, one child, and all
    /// children, because all of these configurations hit special code paths in
    /// the group constructor function.
    fn child_subset(
        children_ids: Vec<TopologyObjectID>,
    ) -> impl Strategy<Value = HashSet<TopologyObjectID>> {
        // Absence of children hits a proptest edge case (can't pick one
        // element from and empty array) and must be handled separately
        if children_ids.is_empty() {
            return Just(HashSet::new()).boxed();
        }

        // Other cases are handled as appropriate
        let num_children = children_ids.len();
        let no_children = Just(HashSet::new());
        let single_child = prop::sample::select(children_ids.clone())
            .prop_map(|child| std::iter::once(child).collect::<HashSet<_>>());
        let some_children = prop::sample::subsequence(children_ids.clone(), 1..=num_children)
            .prop_map(|children| children.into_iter().collect::<HashSet<_>>());
        let all_children = Just(children_ids.into_iter().collect::<HashSet<_>>());
        prop_oneof![
            1 => prop_oneof![no_children, all_children],
            1 => single_child,
            3 => some_children,
        ]
        .boxed()
    }

    /// Turn a child subset into a child-filtering function
    fn filter_fn(subset: HashSet<TopologyObjectID>) -> DynChildFilter {
        Box::new(move |obj| subset.contains(&obj.global_persistent_index()))
    }

    /// If an object belonged to some initial topology, find the equivalent in a
    /// copy of that initial topology (that may be modified, but not in a way
    /// that deletes the parent), otherwise return the parent object as-is
    fn find_parent_like(
        initial_topology: &Topology,
        parent: &'static TopologyObject,
    ) -> impl FnMut(&Topology) -> &TopologyObject {
        let valid_parent_info = initial_topology
            .contains(parent)
            .then(|| (parent.depth(), parent.global_persistent_index()));
        move |copied_topology| {
            if let Some((depth, id)) = valid_parent_info {
                // If the parent belonged to the initial topology,
                // find the equivalent in the copied topology
                copied_topology
                    .objects_at_depth(depth)
                    .find(|obj| obj.global_persistent_index() == id)
                    .expect("parent should still be present in copied_topology")
            } else {
                // Foreign parent of initial topology is also
                // foreign to new topology
                parent
            }
        }
    }

    proptest! {
        /// General-case test for [`TopologyEditor::insert_group_object()`]
        ///
        /// Some specific aspects of this function are not well handled by this
        /// test, but they are stressed by other tests below.
        #[test]
        fn insert_group_object(
            (parent, mut child_filter, dont_merge) in group_building_blocks(),
        ) {
            let initial_topology = Topology::test_instance();
            assert_ne!(
                initial_topology.type_filter(ObjectType::Group).unwrap(),
                TypeFilter::KeepNone,
            );
            let initial_children = child_filter.filter_children(parent, false);
            let children_ids = initial_children
                .as_ref()
                .map(|children| {
                    children
                        .iter()
                        .map(|child| child.global_persistent_index())
                        .collect::<HashSet<_>>()
                })
                .map_err(Clone::clone);
            let mut topology = initial_topology.clone();
            topology.edit(move |editor| {
                let result = editor.insert_group_object(
                    dont_merge,
                    find_parent_like(initial_topology, parent),
                    child_filter,
                );

                // Parent must be a normal object
                if !parent.object_type().is_normal() {
                    prop_assert_eq!(
                        result.unwrap_err(),
                        HybridError::Rust(InsertGroupError::BadParentType(parent.object_type()))
                    );
                    prop_assert_eq!(editor.topology(), initial_topology);
                    return Ok(());
                }

                // Parent must belong to the topology
                if !initial_topology.contains(parent) {
                    prop_assert_eq!(
                        result.unwrap_err(),
                        HybridError::Rust(InsertGroupError::ForeignParent(parent.into()))
                    );
                    prop_assert_eq!(editor.topology(), initial_topology);
                    return Ok(());
                }

                // Group must have at least one child
                if children_ids == Ok(HashSet::new()) {
                    prop_assert_eq!(result.unwrap_err(), HybridError::Rust(InsertGroupError::Empty));
                    prop_assert_eq!(editor.topology(), initial_topology);
                    return Ok(());
                }

                // Group child set must be consistent
                let Ok(children_ids) = children_ids else {
                    prop_assert_eq!(children_ids, Err(InsertGroupError::Inconsistent));
                    prop_assert_eq!(result.unwrap_err(), HybridError::Rust(InsertGroupError::Inconsistent));
                    prop_assert_eq!(editor.topology(), initial_topology);
                    return Ok(());
                };
                let initial_children = initial_children.unwrap();

                // All error paths have been considered, at this point we know
                // that group creation should succeed
                let result = result.unwrap();

                // Now we must handle node equivalence and group merging, which
                // can happen...
                // - If a single child is selected
                // - If all children were selected and the complete_cpuset and
                //   complete_nodeset of the parent do not disambiguate.
                //
                // In all cases, we need to be able to tell which topology nodes
                // are equivalent to each other in hwloc's eyes...
                fn equivalent_obj_ids(mut obj: &TopologyObject) -> HashSet<TopologyObjectID> {
                    let is_equivalent = |candidate: &TopologyObject| {
                        candidate.cpuset() == obj.cpuset()
                            && candidate.nodeset() == obj.nodeset()
                            && candidate.complete_cpuset() == obj.complete_cpuset()
                            && candidate.complete_nodeset() == obj.complete_nodeset()
                    };
                    while obj.normal_arity() == 1 {
                        let only_child = obj.normal_children().next().unwrap();
                        if is_equivalent(only_child) {
                            obj = only_child;
                        } else {
                            break;
                        }
                    }
                    std::iter::once(obj).chain(obj.ancestors())
                        .take_while(|ancestor| is_equivalent(ancestor))
                        .map(TopologyObject::global_persistent_index)
                        .collect()
                }
                //
                // ...and have a way to handle a group that's equivalent to that
                let mut handle_group_equivalence = |
                    result,
                    equivalent_obj: &TopologyObject
                | {
                    let equivalent_ids = equivalent_obj_ids(equivalent_obj);
                    if dont_merge
                        && !equivalent_ids.contains(&initial_topology.root_object().global_persistent_index())
                    {
                        let InsertedGroup::New(group) = result else { unreachable!() };
                        // New group can be inserted below an existing object...
                        if let Some(parent) = group.parent() {
                            if equivalent_ids.contains(&parent.global_persistent_index()) {
                                return Ok(());
                            }
                        }
                        // ...or above it
                        prop_assert_eq!(group.normal_arity(), 1);
                        let only_child = group.normal_children().next().unwrap();
                        prop_assert!(equivalent_ids.contains(&only_child.global_persistent_index()));
                    } else {
                        // Without GroupMerge::Never, should just point at
                        // existing object
                        prop_assert!(matches!(
                            result,
                            InsertedGroup::Existing(obj)
                                if equivalent_ids.contains(&obj.global_persistent_index())
                        ));
                        prop_assert_eq!(editor.topology(), initial_topology);
                    }
                    Ok(())
                };

                // Single-child case
                if initial_children.len() == 1 {
                    handle_group_equivalence(result, initial_children[0])?;
                    return Ok(());
                }

                // Parent-equivalent case
                let parent_sets = (
                    parent.cpuset().unwrap().clone_target(),
                    parent.nodeset().unwrap().clone_target(),
                    parent.complete_cpuset().unwrap().clone_target(),
                    parent.complete_nodeset().unwrap().clone_target(),
                );
                let children_sets_union = initial_children.iter().fold(
                    (CpuSet::new(), NodeSet::new(), CpuSet::new(), NodeSet::new()),
                    |(mut cpuset, mut nodeset, mut complete_cpuset, mut complete_nodeset), child| {
                        cpuset |= child.cpuset().unwrap();
                        nodeset |= child.nodeset().unwrap();
                        complete_cpuset |= child.complete_cpuset().unwrap();
                        complete_nodeset |= child.complete_nodeset().unwrap();
                        (cpuset, nodeset, complete_cpuset, complete_nodeset)
                    }
                );
                if parent_sets == children_sets_union {
                    handle_group_equivalence(result, parent)?;
                    return Ok(());
                }

                // Outside of the conditions enumerated above, a new group
                // should have been created, with the expected set of children
                let InsertedGroup::New(group) = result else { unreachable!() };
                prop_assert_eq!(
                    group.parent().unwrap().global_persistent_index(),
                    parent.global_persistent_index()
                );
                let mut remaining_children_ids = children_ids;
                for child in group.normal_children().chain(group.memory_children()) {
                    prop_assert!(remaining_children_ids.remove(&child.global_persistent_index()));
                }
                prop_assert!(remaining_children_ids.is_empty());
                Ok(())
            })?;
        }

        /// Test that group insertion fails when group type filter is KeepNone
        #[test]
        fn ignored_group_insertion(
            (parent, child_filter, dont_merge) in group_building_blocks(),
        ) {
            static INITIAL_TOPOLOGY: OnceLock<Topology> = OnceLock::new();
            let initial_topology = INITIAL_TOPOLOGY.get_or_init(|| {
                Topology::builder()
                    .with_type_filter(ObjectType::Group, TypeFilter::KeepNone).unwrap()
                    .build().unwrap()
            });

            let mut topology = initial_topology.clone();
            topology.edit(move |editor| {
                let result = editor.insert_group_object(
                    dont_merge,
                    |topology| {
                        if initial_topology.contains(parent) {
                            // If the parent belonged to the initial topology,
                            // find the equivalent in the copied topology
                            topology
                                .objects_at_depth(parent.depth())
                                .find(|obj| obj.global_persistent_index() == parent.global_persistent_index())
                                .expect("parent was tested to be present")
                        } else {
                            // Foreign parent of initial topology is also
                            // foreign to new topology
                            parent
                        }
                    },
                    child_filter,
                );
                prop_assert_eq!(
                    result.unwrap_err(),
                    HybridError::Rust(InsertGroupError::FilteredOut)
                );
                prop_assert_eq!(editor.topology(), initial_topology);
                Ok(())
            })?;
        }
    }

    // --- Misc objects ---

    /// General test of misc object insertion
    fn check_insert_misc_object(
        initial_topology: &Topology,
        name: &str,
        parent: &'static TopologyObject,
    ) -> Result<Topology, TestCaseError> {
        // Check if a misc object with this name already exists
        let name_already_exists = initial_topology
            .objects_with_type(ObjectType::Misc)
            .any(|obj| {
                let Some(obj_name) = obj.name() else {
                    return false;
                };
                let Ok(obj_name) = obj_name.to_str() else {
                    return false;
                };
                obj_name == name
            });

        // Attempt to insert a misc object
        let mut topology = initial_topology.clone();
        topology.edit(|editor| {
            let res = editor
                .insert_misc_object(name, find_parent_like(Topology::test_instance(), parent));

            // Make sure Misc objects aren't filtered out
            let topology = editor.topology();
            if topology.type_filter(ObjectType::Misc).unwrap() == TypeFilter::KeepNone {
                prop_assert_eq!(
                    res.unwrap_err(),
                    HybridError::Rust(InsertMiscError::FilteredOut)
                );
                assert_eq!(topology, initial_topology);
                return Ok(());
            }

            // Make sure no object with this name already exists
            if name_already_exists {
                prop_assert_eq!(
                    res.unwrap_err(),
                    HybridError::Rust(InsertMiscError::NameAlreadyExists)
                );
                assert_eq!(topology, initial_topology);
                return Ok(());
            }

            // Make sure the parent does belong to this topology
            let mut find_parent = find_parent_like(Topology::test_instance(), parent);
            let parent = find_parent(topology);
            if !topology.contains(parent) {
                prop_assert_eq!(
                    res.unwrap_err(),
                    HybridError::Rust(InsertMiscError::ForeignParent(parent.into()))
                );
                assert_eq!(topology, initial_topology);
                return Ok(());
            }

            // Make sure the object name doesn't contain NUL chars
            if name.contains('\0') {
                prop_assert_eq!(res.unwrap_err(), HybridError::Rust(NulError.into()));
                assert_eq!(topology, initial_topology);
                return Ok(());
            }

            // If all of the above passed, creation should succeed
            let obj = res.unwrap();
            prop_assert_eq!(obj.object_type(), ObjectType::Misc);
            prop_assert_eq!(obj.name().unwrap().to_str().unwrap(), name);
            prop_assert!(parent
                .misc_children()
                .any(|child| child.global_persistent_index() == obj.global_persistent_index()));
            Ok(())
        })?;
        Ok(topology)
    }

    proptest! {
        /// ...with the normal type filter
        #[test]
        fn insert_misc_object(
            name in any_string(),
            parent in any_object(),
        ) {
            check_insert_misc_object(Topology::test_instance(), &name, parent)?;
        }

        /// ...with a type filter that filters out Misc objects
        #[test]
        fn ignored_misc_insertion(
            name in any_string(),
            parent in any_object(),
        ) {
            static INITIAL_TOPOLOGY: OnceLock<Topology> = OnceLock::new();
            let initial_topology = INITIAL_TOPOLOGY.get_or_init(|| {
                Topology::builder()
                    .with_type_filter(ObjectType::Misc, TypeFilter::KeepNone).unwrap()
                    .build().unwrap()
            });
            check_insert_misc_object(initial_topology, &name, parent)?;
        }

        /// ...twice with the same name, which should error out
        #[test]
        fn duplicate(
            name in any_string(),
            (parent1, parent2) in (any_object(), any_object()),
        ) {
            let topology = check_insert_misc_object(Topology::test_instance(), &name, parent1)?;
            check_insert_misc_object(&topology, &name, parent2)?;
        }

        /// ...twice with separate names, which may succeed
        #[test]
        fn separate(
            (name1, name2) in (any_string(), any_string()),
            (parent1, parent2) in (any_object(), any_object()),
        ) {
            let topology = check_insert_misc_object(Topology::test_instance(), &name1, parent1)?;
            check_insert_misc_object(&topology, &name2, parent2)?;
        }
    }
}
