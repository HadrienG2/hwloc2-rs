//! Facilities for manipulating bitmaps
//!
//! # Bitmaps
//!
//! The hwloc library extensively uses bitmaps to model the concept of sets of
//! CPU cores that threads and processes can be bound to, and that of sets of
//! NUMA nodes that memory allocations can be bound to.
//!
//! In this module, we directly re-export this API via the [`Bitmap`] type.
//! However, other methods of the hwlocality binding do not directly accept or
//! emit bitmaps, as they do in the underlying hwloc C library. Instead they
//! use specialized variants of [`Bitmap`] called [`CpuSet`] and [`NodeSet`].
//!
//! These types are trivial wrappers around [`Bitmap`] that have basically the
//! same API, but provide improved type safety (you cannot use a
//! [`NodeSet`] where a [`CpuSet`] is expected, nor can you mistakenly mix CPU
//! indices with NUMA node indices). They also make the few APIs that accept
//! either cpusets or nodesets easier to use with respect to upstream hwloc, as
//! you don't need to remember to set or clear flags depending on which one of
//! these you are passing to the function.
//!
//! # Bitmap indices
//!
//! The range of indices that can be set or cleared in an hwloc bitmap
//! unfortunately does not neatly map into any machine integer type. It is
//! therefore modeled via the dedicated [`BitmapIndex`] integer type, which you
//! can use for optimal type safety, at the expense of facing some awkward API
//! ergonomics since user-defined integer types unfortunately cannot perfectly
//! match the ergonomics of machine types in Rust (for one thing, they don't
//! have literals).
//!
//! If this proves to be too cumbersome for you, an escape hatch is provided so
//! that you can easily use [`usize`] in almost every place where
//! [`BitmapIndex`] is expected, at the cost of risking a panic if the
//! provided [`usize`] is out of the [`BitmapIndex`] range. This design puts you
//! in control of the underlying compromise between type safety and ergonomics,
//! letting you choose what works best for you depending on your requirements.
//!
//! # Bitmap references and polymorphism
//!
//! For obscure implementation reasons, references to bitmaps that are owned by
//! hwloc cannot be provided as simple `&'target Target` references. Instead, a
//! dedicated [`BitmapRef`] type which behaves analogously must be used.
//!
//! For convenience, many hwlocality methods achieve genericity with respect to
//! [`BitmapRef`] and standard Rust references to the specialized [`CpuSet`] and
//! [`NodeSet`] bitmap types by leveraging the [`SpecializedBitmapRef`] trait.
//
// Main docs: https://hwloc.readthedocs.io/en/v2.9/group__hwlocality__bitmap.html

mod cow;
mod newtypes;
mod reference;

#[cfg(doc)]
use crate::{
    cpu::cpuset::CpuSet, memory::nodeset::NodeSet, object::TopologyObject, topology::Topology,
};
use crate::{
    errors,
    ffi::{self, PositiveInt},
};
use hwlocality_sys::hwloc_bitmap_s;
#[cfg(any(test, feature = "proptest"))]
use proptest::prelude::*;
#[allow(unused)]
#[cfg(test)]
use similar_asserts::assert_eq;
#[cfg(any(test, feature = "proptest"))]
use std::collections::HashSet;
use std::{
    borrow::Borrow,
    cmp::Ordering,
    ffi::{c_int, c_uint},
    fmt::{self, Debug, Display, Formatter, Pointer},
    hash::{self, Hash},
    iter::FusedIterator,
    ops::{
        BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Bound, Deref, Not,
        RangeBounds, Sub, SubAssign,
    },
    ptr::NonNull,
};

/// Valid bitmap index ranging from `0` to [`c_int::MAX`]
pub type BitmapIndex = PositiveInt;

// Re-export BitmapRef so users don't need to know about the reference submodule
pub use self::{
    cow::BitmapCow,
    newtypes::{BitmapKind, OwnedBitmap, SpecializedBitmap, SpecializedBitmapRef},
    reference::BitmapRef,
};

/// A generic bitmap, understood by hwloc
///
/// The `Bitmap` type represents a set of integers (positive or null). A bitmap
/// may be of infinite size (all bits are set after some point). A bitmap may
/// even be full if all bits are set.
///
/// Bitmaps are used by hwloc to represent sets of OS processors (which may
/// actually be hardware threads), via the [`CpuSet`] newtype, or to represent
/// sets of NUMA memory nodes via the [`NodeSet`] newtype.
///
/// Both [`CpuSet`] and [`NodeSet`] are always indexed by OS physical number.
/// However, users should usually not build CPU and node sets manually (e.g.
/// with [`Bitmap::set()`]). One should rather use the cpu/node sets of existing
/// [`TopologyObject`]s and combine them through boolean operations. For
/// instance, binding the current thread on a pair of cores may be performed
/// with:
///
/// ```
/// # use hwlocality::{
/// #     cpu::{binding::CpuBindingFlags, cpuset::CpuSet},
/// #     object::{types::ObjectType},
/// #     topology::support::{CpuBindingSupport, FeatureSupport},
/// # };
/// #
/// # let topology = hwlocality::Topology::test_instance();
/// #
/// // We want Cores, but we tolerate PUs on platforms that don't expose Cores
/// // (either there are no hardware threads or hwloc could not detect them)
/// let core_depth = topology.depth_or_below_for_type(ObjectType::Core)?;
///
/// // Yields the first two cores of a multicore system, or
/// // the only core of a single-core system
/// let cores = topology.objects_at_depth(core_depth).take(2);
///
/// // Compute the union of these cores' CPUsets, that's our CPU binding bitmap
/// let set = cores.fold(
///     CpuSet::new(),
///     |acc, core| { acc | core.cpuset().expect("Cores should have CPUsets") }
/// );
///
/// // Only actually bind if the platform supports it (macOS notably doesn't)
/// if topology.supports(FeatureSupport::cpu_binding, CpuBindingSupport::set_thread) {
///     topology.bind_cpu(&set, CpuBindingFlags::THREAD)?;
/// }
/// #
/// # Ok::<(), eyre::Report>(())
/// ```
///
/// # Panics
///
/// Unlike most hwloc entry points in this crate, `Bitmap` functions always
/// handle unexpected hwloc errors by panicking. The rationale for this is that
/// the bitmap is just a simple data structures, without any kind of
/// complicated interactions with the operating system, for which the only
/// failure mode should be running out of memory. And panicking is the normal
/// way to handle this in Rust.
///
/// [`CpuSet`]: crate::cpu::cpuset::CpuSet
/// [`NodeSet`]: crate::memory::nodeset::NodeSet
//
// --- Implementation details ---
//
// # Safety
//
// - As a type invariant, the inner pointer is assumed to always point to a
//   valid, non-aliased bitmap
// - &mut self should only be exposed if the bitmap is safe to modify
// - &mut self or owned Self should only be exposed if the bitmap is safe to
//   drop (i.e. not topology-owned)
#[doc(alias = "hwloc_bitmap_t")]
#[doc(alias = "hwloc_const_bitmap_t")]
#[repr(transparent)]
pub struct Bitmap(NonNull<hwloc_bitmap_s>);

// NOTE: Remember to keep the method signatures and first doc lines in
//       impl_newtype_ops in sync with what's going on below.
impl Bitmap {
    // === FFI interoperability ===

    /// Wraps an owned nullable hwloc bitmap
    ///
    /// # Safety
    ///
    /// If non-null, the pointer must target a valid bitmap that we will acquire
    /// ownership of and automatically free on `Drop`. This bitmap must
    /// therefore be safe to free, it should not belong to a topology.
    pub(crate) unsafe fn from_owned_raw_mut(bitmap: *mut hwloc_bitmap_s) -> Option<Self> {
        // SAFETY: Per function input precondition
        NonNull::new(bitmap).map(|ptr| unsafe { Self::from_owned_nonnull(ptr) })
    }

    /// Wraps an owned hwloc bitmap
    ///
    /// # Safety
    ///
    /// The pointer must target a valid bitmap that we will acquire ownership of
    /// and automatically free on `Drop`. This bitmap must therefore be safe to
    /// free, it should not belong to a topology.
    pub(crate) unsafe fn from_owned_nonnull(bitmap: NonNull<hwloc_bitmap_s>) -> Self {
        // SAFETY: Per function input precondition
        Self(bitmap)
    }

    /// Wraps a borrowed nullable hwloc bitmap
    ///
    /// # Safety
    ///
    /// If non-null, the pointer must target a bitmap that is valid for
    /// `'target`. Unlike with [`Bitmap::from_owned_raw_mut()`], it will not be
    /// automatically freed on `Drop`.
    pub(crate) unsafe fn borrow_from_raw<'target>(
        bitmap: *const hwloc_bitmap_s,
    ) -> Option<BitmapRef<'target, Self>> {
        // SAFETY: Per function input precondition
        unsafe { Self::borrow_from_raw_mut(bitmap.cast_mut()) }
    }

    /// Wraps a borrowed nullable hwloc bitmap
    ///
    /// # Safety
    ///
    /// If non-null, the pointer must target a bitmap that is valid for
    /// `'target`. Unlike with [`Bitmap::from_owned_raw_mut()`], it will not be
    /// automatically freed on `Drop`.
    pub(crate) unsafe fn borrow_from_raw_mut<'target>(
        bitmap: *mut hwloc_bitmap_s,
    ) -> Option<BitmapRef<'target, Self>> {
        // SAFETY: Per function input precondition
        NonNull::new(bitmap).map(|ptr| unsafe { Self::borrow_from_nonnull(ptr) })
    }

    /// Wraps a borrowed hwloc bitmap
    ///
    /// # Safety
    ///
    /// The pointer must target a bitmap that is valid for `'target`. Unlike
    /// with [`Bitmap::from_owned_nonnull()`], it will not be automatically
    /// freed on `Drop`.
    pub(crate) unsafe fn borrow_from_nonnull<'target>(
        bitmap: NonNull<hwloc_bitmap_s>,
    ) -> BitmapRef<'target, Self> {
        // SAFETY: Per function input precondition
        unsafe { BitmapRef::from_nonnull(bitmap) }
    }

    // NOTE: There is no borrow_mut_from_raw because it would not be safe as if
    //       you expose an &mut Bitmap, the user can trigger Drop.

    /// Contained bitmap pointer (for interaction with hwloc)
    pub(crate) fn as_ptr(&self) -> *const hwloc_bitmap_s {
        self.0.as_ptr()
    }

    /// Contained mutable bitmap pointer (for interaction with hwloc)
    pub(crate) fn as_mut_ptr(&mut self) -> *mut hwloc_bitmap_s {
        self.0.as_ptr()
    }

    // === Constructors ===

    /// Creates an empty `Bitmap`
    ///
    /// # Examples
    ///
    /// ```
    /// use hwlocality::bitmap::Bitmap;
    ///
    /// let empty = Bitmap::new();
    /// assert!(empty.is_empty());
    /// ```
    #[doc(alias = "hwloc_bitmap_alloc")]
    pub fn new() -> Self {
        // SAFETY: - This function has no safety postcondition
        //         - It is trusted to produce valid owned bitmaps
        unsafe {
            let ptr = errors::call_hwloc_ptr_mut("hwloc_bitmap_alloc", || {
                hwlocality_sys::hwloc_bitmap_alloc()
            })
            .expect(MALLOC_FAIL_ONLY);
            Self::from_owned_nonnull(ptr)
        }
    }

    /// Creates a full `Bitmap`
    ///
    /// # Examples
    ///
    /// ```
    /// use hwlocality::bitmap::Bitmap;
    ///
    /// let full = Bitmap::full();
    /// assert!(full.is_full());
    /// ```
    #[doc(alias = "hwloc_bitmap_alloc_full")]
    pub fn full() -> Self {
        // SAFETY: - This function has no safety postcondition
        //         - It is trusted to produce valid owned bitmaps
        unsafe {
            let ptr = errors::call_hwloc_ptr_mut("hwloc_bitmap_alloc_full", || {
                hwlocality_sys::hwloc_bitmap_alloc_full()
            })
            .expect(MALLOC_FAIL_ONLY);
            Self::from_owned_nonnull(ptr)
        }
    }

    /// Creates a new `Bitmap` with the given range of indices set
    ///
    /// Accepts both ranges of [`BitmapIndex`] and [`usize`]. Use the former for
    /// type safety (it is guaranteed to be in range as a type invariant) or the
    /// latter for convenience (it is more tightly integrated with Rust's
    /// built-in integer support, for example it supports integer literals).
    ///
    /// # Examples
    ///
    /// ```
    /// use hwlocality::bitmap::Bitmap;
    ///
    /// let bitmap = Bitmap::from_range(12..=34);
    /// assert_eq!(format!("{bitmap}"), "12-34");
    /// ```
    ///
    /// # Panics
    ///
    /// If `range` goes beyond the implementation-defined maximum index (at
    /// least 2^15-1, usually 2^31-1).
    pub fn from_range<Idx>(range: impl RangeBounds<Idx>) -> Self
    where
        Idx: Copy + TryInto<BitmapIndex>,
        <Idx as TryInto<BitmapIndex>>::Error: Debug,
    {
        let mut bitmap = Self::new();
        bitmap.set_range(range);
        bitmap
    }

    // === Getters and setters ===

    /// Turn this `Bitmap` into a copy of another `Bitmap`
    ///
    /// Accepts both `&'_ Bitmap` and `BitmapRef<'_, Bitmap>` operands.
    ///
    /// # Examples
    ///
    /// ```
    /// use hwlocality::bitmap::Bitmap;
    ///
    /// let bitmap = Bitmap::from_range(12..=34);
    /// let mut bitmap2 = Bitmap::new();
    /// bitmap2.copy_from(&bitmap);
    /// assert_eq!(format!("{bitmap2}"), "12-34");
    /// ```
    #[doc(alias = "hwloc_bitmap_copy")]
    pub fn copy_from(&mut self, other: impl Deref<Target = Self>) {
        /// Polymorphized version of this function (avoids generics code bloat)
        fn polymorphized(self_: &mut Bitmap, other: &Bitmap) {
            // SAFETY: - Bitmaps are trusted to contain a valid ptr (type invariant)
            //         - hwloc ops are trusted not to modify *const parameters
            //         - hwloc ops are trusted to keep *mut parameters in a
            //           valid state unless stated otherwise
            errors::call_hwloc_int_normal("hwloc_bitmap_copy", || unsafe {
                hwlocality_sys::hwloc_bitmap_copy(self_.as_mut_ptr(), other.as_ptr())
            })
            .expect(MALLOC_FAIL_ONLY);
        }
        polymorphized(self, &other)
    }

    /// Clear all indices
    ///
    /// # Examples
    ///
    /// ```
    /// use hwlocality::bitmap::Bitmap;
    ///
    /// let mut bitmap = Bitmap::from_range(12..=34);
    /// bitmap.clear();
    /// assert!(bitmap.is_empty());
    /// ```
    #[doc(alias = "hwloc_bitmap_zero")]
    pub fn clear(&mut self) {
        // SAFETY: - Bitmaps are trusted to contain a valid ptr (type invariant)
        //         - hwloc ops are trusted to keep *mut parameters in a
        //           valid state unless stated otherwise
        unsafe { hwlocality_sys::hwloc_bitmap_zero(self.as_mut_ptr()) }
    }

    /// Set all indices
    ///
    /// # Examples
    ///
    /// ```
    /// use hwlocality::bitmap::Bitmap;
    ///
    /// let mut bitmap = Bitmap::from_range(12..=34);
    /// bitmap.fill();
    /// assert!(bitmap.is_full());
    /// ```
    #[doc(alias = "hwloc_bitmap_fill")]
    pub fn fill(&mut self) {
        // SAFETY: - Bitmaps are trusted to contain a valid ptr (type invariant)
        //         - hwloc ops are trusted to keep *mut parameters in a
        //           valid state unless stated otherwise
        unsafe { hwlocality_sys::hwloc_bitmap_fill(self.as_mut_ptr()) }
    }

    /// Clear all indices except for `idx`, which is set
    ///
    /// Accepts both [`BitmapIndex`] and [`usize`] operands. Use the former for
    /// type-safety (it is guaranteed to be in range as a type invariant) or the
    /// latter for convenience (it is more tightly integrated with Rust's
    /// built-in integer support, for example it supports integer literals).
    ///
    /// # Examples
    ///
    /// ```
    /// use hwlocality::bitmap::Bitmap;
    ///
    /// let mut bitmap = Bitmap::from_range(12..=34);
    /// bitmap.set_only(42);
    /// assert_eq!(format!("{bitmap}"), "42");
    /// ```
    ///
    /// # Panics
    ///
    /// If `idx` is above the implementation-defined maximum index (at least
    /// 2^15-1, usually 2^31-1).
    #[doc(alias = "hwloc_bitmap_only")]
    pub fn set_only<Idx>(&mut self, idx: Idx)
    where
        Idx: TryInto<BitmapIndex>,
        <Idx as TryInto<BitmapIndex>>::Error: Debug,
    {
        /// Polymorphized version of this function (avoids generics code bloat)
        fn polymorphized(self_: &mut Bitmap, idx: Option<BitmapIndex>) {
            let idx = idx.expect(BAD_INDEX);
            // SAFETY: - Bitmaps are trusted to contain a valid ptr (type invariant)
            //         - idx has been checked to be in the hwloc-supported range
            //         - hwloc ops are trusted to keep *mut parameters in a
            //           valid state unless stated otherwise
            errors::call_hwloc_int_normal("hwloc_bitmap_only", || unsafe {
                hwlocality_sys::hwloc_bitmap_only(self_.as_mut_ptr(), idx.to_c_uint())
            })
            .expect(MALLOC_FAIL_ONLY);
        }
        polymorphized(self, idx.try_into().ok());
    }

    /// Set all indices except for `idx`, which is cleared
    ///
    /// Accepts both [`BitmapIndex`] and [`usize`] operands. Use the former for
    /// type-safety (it is guaranteed to be in range as a type invariant) or the
    /// latter for convenience (it is more tightly integrated with Rust's
    /// built-in integer support, for example it supports integer literals).
    ///
    /// # Examples
    ///
    /// ```
    /// use hwlocality::bitmap::Bitmap;
    ///
    /// let mut bitmap = Bitmap::from_range(12..=34);
    /// bitmap.set_all_but(42);
    /// assert_eq!(format!("{bitmap}"), "0-41,43-");
    /// ```
    ///
    /// # Panics
    ///
    /// If `idx` is above the implementation-defined maximum index (at least
    /// 2^15-1, usually 2^31-1).
    #[doc(alias = "hwloc_bitmap_allbut")]
    pub fn set_all_but<Idx>(&mut self, idx: Idx)
    where
        Idx: TryInto<BitmapIndex>,
        <Idx as TryInto<BitmapIndex>>::Error: Debug,
    {
        /// Polymorphized version of this function (avoids generics code bloat)
        fn polymorphized(self_: &mut Bitmap, idx: Option<BitmapIndex>) {
            let idx = idx.expect(BAD_INDEX);
            // SAFETY: - Bitmaps are trusted to contain a valid ptr (type invariant)
            //         - idx has been checked to be in the hwloc-supported range
            //         - hwloc ops are trusted to keep *mut parameters in a
            //           valid state unless stated otherwise
            errors::call_hwloc_int_normal("hwloc_bitmap_allbut", || unsafe {
                hwlocality_sys::hwloc_bitmap_allbut(self_.as_mut_ptr(), idx.to_c_uint())
            })
            .expect(MALLOC_FAIL_ONLY);
        }
        polymorphized(self, idx.try_into().ok());
    }

    /// Set index `idx`
    ///
    /// Accepts both [`BitmapIndex`] and [`usize`] operands. Use the former for
    /// type-safety (it is guaranteed to be in range as a type invariant) or the
    /// latter for convenience (it is more tightly integrated with Rust's
    /// built-in integer support, for example it supports integer literals).
    ///
    /// # Examples
    ///
    /// ```
    /// use hwlocality::bitmap::Bitmap;
    ///
    /// let mut bitmap = Bitmap::from_range(12..=34);
    /// bitmap.set(42);
    /// assert_eq!(format!("{bitmap}"), "12-34,42");
    /// ```
    ///
    /// # Panics
    ///
    /// If `idx` is above the implementation-defined maximum index (at least
    /// 2^15-1, usually 2^31-1).
    #[doc(alias = "hwloc_bitmap_set")]
    pub fn set<Idx>(&mut self, idx: Idx)
    where
        Idx: TryInto<BitmapIndex>,
        <Idx as TryInto<BitmapIndex>>::Error: Debug,
    {
        /// Polymorphized version of this function (avoids generics code bloat)
        fn polymorphized(self_: &mut Bitmap, idx: Option<BitmapIndex>) {
            let idx = idx.expect(BAD_INDEX);
            // SAFETY: - Bitmaps are trusted to contain a valid ptr (type invariant)
            //         - idx has been checked to be in the hwloc-supported range
            //         - hwloc ops are trusted to keep *mut parameters in a
            //           valid state unless stated otherwise
            errors::call_hwloc_int_normal("hwloc_bitmap_set", || unsafe {
                hwlocality_sys::hwloc_bitmap_set(self_.as_mut_ptr(), idx.to_c_uint())
            })
            .expect(MALLOC_FAIL_ONLY);
        }
        polymorphized(self, idx.try_into().ok());
    }

    /// Set indices covered by `range`
    ///
    /// Accepts both ranges of [`BitmapIndex`] and [`usize`]. Use the former for
    /// type-safety (it is guaranteed to be in range as a type invariant) or the
    /// latter for convenience (it is more tightly integrated with Rust's
    /// built-in integer support, for example it supports integer literals).
    ///
    /// # Examples
    ///
    /// ```
    /// use hwlocality::bitmap::Bitmap;
    ///
    /// let mut bitmap = Bitmap::from_range(12..=56);
    /// bitmap.set_range(34..=78);
    /// assert_eq!(format!("{bitmap}"), "12-78");
    ///
    /// bitmap.set_range(2..);
    /// assert_eq!(format!("{bitmap}"), "2-");
    /// ```
    ///
    /// # Panics
    ///
    /// If `range` goes beyond the implementation-defined maximum index (at
    /// least 2^15-1, usually 2^31-1).
    #[doc(alias = "hwloc_bitmap_set_range")]
    pub fn set_range<Idx>(&mut self, range: impl RangeBounds<Idx>)
    where
        Idx: Copy + TryInto<BitmapIndex>,
        <Idx as TryInto<BitmapIndex>>::Error: Debug,
    {
        /// Polymorphized version of this function (avoids generics code bloat)
        ///
        /// # Safety
        ///
        /// `(begin, end)` must be a pair of valid hwloc range bounds, i.e.
        /// `begin` must not be larger than [`c_int::MAX`] and `end` must not be
        /// smaller than -1.
        unsafe fn polymorphized(self_: &mut Bitmap, (begin, end): (c_uint, c_int)) {
            // SAFETY: - Bitmaps are trusted to contain a valid ptr (type invariant)
            //         - Range bounds are trusted per function precondition
            //         - hwloc ops are trusted to keep *mut parameters in a
            //           valid state unless stated otherwise
            errors::call_hwloc_int_normal("hwloc_bitmap_set_range", || unsafe {
                hwlocality_sys::hwloc_bitmap_set_range(self_.as_mut_ptr(), begin, end)
            })
            .expect(MALLOC_FAIL_ONLY);
        }
        // SAFETY: The output of hwloc_range is trusted
        unsafe { polymorphized(self, Self::hwloc_range(range)) };
    }

    /// Clear index `idx`
    ///
    /// Accepts both [`BitmapIndex`] and [`usize`] operands. Use the former for
    /// type-safety (it is guaranteed to be in range as a type invariant) or the
    /// latter for convenience (it is more tightly integrated with Rust's
    /// built-in integer support, for example it supports integer literals).
    ///
    /// # Examples
    ///
    /// ```
    /// use hwlocality::bitmap::Bitmap;
    ///
    /// let mut bitmap = Bitmap::from_range(12..=34);
    /// bitmap.unset(24);
    /// assert_eq!(format!("{bitmap}"), "12-23,25-34");
    /// ```
    ///
    /// # Panics
    ///
    /// If `idx` is above the implementation-defined maximum index (at least
    /// 2^15-1, usually 2^31-1).
    #[doc(alias = "hwloc_bitmap_clr")]
    pub fn unset<Idx>(&mut self, idx: Idx)
    where
        Idx: TryInto<BitmapIndex>,
        <Idx as TryInto<BitmapIndex>>::Error: Debug,
    {
        /// Polymorphized version of this function (avoids generics code bloat)
        fn polymorphized(self_: &mut Bitmap, idx: Option<BitmapIndex>) {
            let idx = idx.expect(BAD_INDEX);
            // SAFETY: - Bitmaps are trusted to contain a valid ptr (type invariant)
            //         - idx has been checked to be in the hwloc-supported range
            //         - hwloc ops are trusted to keep *mut parameters in a
            //           valid state unless stated otherwise
            errors::call_hwloc_int_normal("hwloc_bitmap_clr", || unsafe {
                hwlocality_sys::hwloc_bitmap_clr(self_.as_mut_ptr(), idx.to_c_uint())
            })
            .expect(MALLOC_FAIL_ONLY);
        }
        polymorphized(self, idx.try_into().ok());
    }

    /// Clear indices covered by `range`
    ///
    /// Accepts both ranges of [`BitmapIndex`] and [`usize`]. Use the former for
    /// type-safety (it is guaranteed to be in range as a type invariant) or the
    /// latter for convenience (it is more tightly integrated with Rust's
    /// built-in integer support, for example it supports integer literals).
    ///
    /// # Examples
    ///
    /// ```
    /// use hwlocality::bitmap::Bitmap;
    ///
    /// let mut bitmap = Bitmap::from_range(12..=34);
    /// bitmap.unset_range(14..=18);
    /// assert_eq!(format!("{bitmap}"), "12-13,19-34");
    ///
    /// bitmap.unset_range(26..);
    /// assert_eq!(format!("{bitmap}"), "12-13,19-25");
    /// ```
    ///
    /// # Panics
    ///
    /// If `range` goes beyond the implementation-defined maximum index (at
    /// least 2^15-1, usually 2^31-1).
    #[doc(alias = "hwloc_bitmap_clr_range")]
    pub fn unset_range<Idx>(&mut self, range: impl RangeBounds<Idx>)
    where
        Idx: Copy + TryInto<BitmapIndex>,
        <Idx as TryInto<BitmapIndex>>::Error: Debug,
    {
        /// Polymorphized version of this function (avoids generics code bloat)
        ///
        /// # Safety
        ///
        /// `(begin, end)` must be a pair of valid hwloc range bounds, i.e.
        /// `begin` must not be larger than [`c_int::MAX`] and `end` must not be
        /// smaller than -1.
        unsafe fn polymorphized(self_: &mut Bitmap, (begin, end): (c_uint, c_int)) {
            // SAFETY: - Bitmaps are trusted to contain a valid ptr (type invariant)
            //         - Range bounds are trusted per function precondition
            //         - hwloc ops are trusted to keep *mut parameters in a
            //           valid state unless stated otherwise
            errors::call_hwloc_int_normal("hwloc_bitmap_clr_range", || unsafe {
                hwlocality_sys::hwloc_bitmap_clr_range(self_.as_mut_ptr(), begin, end)
            })
            .expect(MALLOC_FAIL_ONLY);
        }
        // SAFETY: The output of hwloc_range is trusted
        unsafe { polymorphized(self, Self::hwloc_range(range)) };
    }

    /// Keep a single index among those set in the bitmap
    ///
    /// May be useful before binding so that the process does not have a
    /// chance of migrating between multiple logical CPUs in the original mask.
    /// Instead of running the task on any PU inside the given CPU set, the
    /// operating system scheduler will be forced to run it on a single of these
    /// PUs. It avoids a migration overhead and cache-line ping-pongs between PUs.
    ///
    /// This function is NOT meant to distribute multiple processes within a
    /// single CPU set. It always return the same single bit when called
    /// multiple times on the same input set. [`Topology::distribute_items()`]
    /// may be used for generating CPU sets to distribute multiple tasks below a
    /// single multi-PU object.
    ///
    /// The effect of singlifying an empty bitmap is not specified by hwloc.
    ///
    /// # Examples
    ///
    /// ```
    /// use hwlocality::bitmap::Bitmap;
    ///
    /// let mut bitmap = Bitmap::from_range(12..=34);
    /// bitmap.singlify();
    /// assert_eq!(bitmap.weight(), Some(1));
    /// ```
    #[doc(alias = "hwloc_bitmap_singlify")]
    pub fn singlify(&mut self) {
        // SAFETY: - Bitmaps are trusted to contain a valid ptr (type invariant)
        //         - hwloc ops are trusted to keep *mut parameters in a
        //           valid state unless stated otherwise
        errors::call_hwloc_int_normal("hwloc_bitmap_singlify", || unsafe {
            hwlocality_sys::hwloc_bitmap_singlify(self.as_mut_ptr())
        })
        .expect(MALLOC_FAIL_ONLY);
    }

    /// Check if index `idx` is set
    ///
    /// Accepts both [`BitmapIndex`] and [`usize`] operands. Use the former for
    /// type-safety (it is guaranteed to be in range as a type invariant) or the
    /// latter for convenience (it is more tightly integrated with Rust's
    /// built-in integer support, for example it supports integer literals).
    ///
    /// # Examples
    ///
    /// ```
    /// use hwlocality::bitmap::Bitmap;
    ///
    /// let mut bitmap = Bitmap::from_range(12..=34);
    /// assert!((0..12).all(|idx| !bitmap.is_set(idx)));
    /// assert!((12..=34).all(|idx| bitmap.is_set(idx)));
    /// assert!(!bitmap.is_set(35));
    /// ```
    ///
    /// # Panics
    ///
    /// If `idx` is above the implementation-defined maximum index (at least
    /// 2^15-1, usually 2^31-1).
    #[doc(alias = "hwloc_bitmap_isset")]
    pub fn is_set<Idx>(&self, idx: Idx) -> bool
    where
        Idx: TryInto<BitmapIndex>,
        <Idx as TryInto<BitmapIndex>>::Error: Debug,
    {
        /// Polymorphized version of this function (avoids generics code bloat)
        fn polymorphized(self_: &Bitmap, idx: Option<BitmapIndex>) -> bool {
            let idx = idx.expect(BAD_INDEX);
            // SAFETY: - Bitmaps are trusted to contain a valid ptr (type invariant)
            //         - idx has been checked to be in the hwloc-supported range
            //         - hwloc ops are trusted to keep *mut parameters in a
            //           valid state unless stated otherwise
            errors::call_hwloc_bool("hwloc_bitmap_isset", || unsafe {
                hwlocality_sys::hwloc_bitmap_isset(self_.as_ptr(), idx.to_c_uint())
            })
            .expect(MALLOC_FAIL_ONLY)
        }
        polymorphized(self, idx.try_into().ok())
    }

    /// Check if all indices are unset
    ///
    /// # Examples
    ///
    /// ```
    /// use hwlocality::bitmap::Bitmap;
    ///
    /// assert!(Bitmap::new().is_empty());
    /// assert!(!Bitmap::from_range(12..=34).is_empty());
    /// assert!(!Bitmap::full().is_empty());
    /// ```
    #[doc(alias = "hwloc_bitmap_iszero")]
    pub fn is_empty(&self) -> bool {
        // SAFETY: - Bitmaps are trusted to contain a valid ptr (type invariant)
        //         - hwloc ops are trusted to keep them in a valid state
        //         - hwloc ops are trusted not to modify *const parameters
        errors::call_hwloc_bool("hwloc_bitmap_iszero", || unsafe {
            hwlocality_sys::hwloc_bitmap_iszero(self.as_ptr())
        })
        .expect(SHOULD_NOT_FAIL)
    }

    /// Check if all indices are set
    ///
    /// # Examples
    ///
    /// ```
    /// use hwlocality::bitmap::Bitmap;
    ///
    /// assert!(!Bitmap::new().is_full());
    /// assert!(!Bitmap::from_range(12..=34).is_full());
    /// assert!(Bitmap::full().is_full());
    /// ```
    #[doc(alias = "hwloc_bitmap_isfull")]
    pub fn is_full(&self) -> bool {
        // SAFETY: - Bitmaps are trusted to contain a valid ptr (type invariant)
        //         - hwloc ops are trusted not to modify *const parameters
        errors::call_hwloc_bool("hwloc_bitmap_isfull", || unsafe {
            hwlocality_sys::hwloc_bitmap_isfull(self.as_ptr())
        })
        .expect(SHOULD_NOT_FAIL)
    }

    /// Check the first set index, if any
    ///
    /// You can iterate over set indices with [`Bitmap::iter_set()`].
    ///
    /// # Examples
    ///
    /// ```
    /// use hwlocality::bitmap::Bitmap;
    ///
    /// let first_set_usize = |b: Bitmap| b.first_set().map(usize::from);
    /// assert_eq!(Bitmap::new().first_set(), None);
    /// assert_eq!(first_set_usize(Bitmap::from_range(12..=34)), Some(12));
    /// assert_eq!(first_set_usize(Bitmap::full()), Some(0));
    /// ```
    #[doc(alias = "hwloc_bitmap_first")]
    pub fn first_set(&self) -> Option<BitmapIndex> {
        self.query_index(
            "hwloc_bitmap_first",
            // SAFETY: - Bitmaps are trusted to contain a valid ptr (type invariant)
            //         - hwloc ops are trusted not to modify *const parameters
            || unsafe { hwlocality_sys::hwloc_bitmap_first(self.as_ptr()) },
        )
    }

    /// Iterate over set indices
    ///
    /// # Examples
    ///
    /// ```
    /// use hwlocality::bitmap::Bitmap;
    ///
    /// let bitmap = Bitmap::from_range(12..=21);
    /// let indices = bitmap.iter_set().map(usize::from).collect::<Vec<_>>();
    /// assert_eq!(indices, &[12, 13, 14, 15, 16, 17, 18, 19, 20, 21]);
    /// ```
    #[doc(alias = "hwloc_bitmap_foreach_begin")]
    #[doc(alias = "hwloc_bitmap_foreach_end")]
    #[doc(alias = "hwloc_bitmap_next")]
    pub fn iter_set(&self) -> Iter<&Self> {
        Iter::new(self, Self::next_set)
    }

    /// Check the last set index, if any
    ///
    /// # Examples
    ///
    /// ```
    /// use hwlocality::bitmap::Bitmap;
    ///
    /// let last_set_usize = |b: Bitmap| b.last_set().map(usize::from);
    /// assert_eq!(Bitmap::new().last_set(), None);
    /// assert_eq!(last_set_usize(Bitmap::from_range(12..=34)), Some(34));
    /// assert_eq!(Bitmap::full().last_set(), None);
    /// ```
    #[doc(alias = "hwloc_bitmap_last")]
    pub fn last_set(&self) -> Option<BitmapIndex> {
        self.query_index(
            "hwloc_bitmap_last",
            // SAFETY: - Bitmaps are trusted to contain a valid ptr (type invariant)
            //         - hwloc ops are trusted not to modify *const parameters
            || unsafe { hwlocality_sys::hwloc_bitmap_last(self.as_ptr()) },
        )
    }

    /// The number of indices that are set in the bitmap
    ///
    /// None means that an infinite number of indices are set.
    ///
    /// # Examples
    ///
    /// ```
    /// use hwlocality::bitmap::Bitmap;
    ///
    /// assert_eq!(Bitmap::new().weight(), Some(0));
    /// assert_eq!(Bitmap::from_range(12..34).weight(), Some(34-12));
    /// assert_eq!(Bitmap::full().weight(), None);
    /// ```
    #[doc(alias = "hwloc_bitmap_weight")]
    pub fn weight(&self) -> Option<usize> {
        let result = errors::call_hwloc_int_raw(
            "hwloc_bitmap_weight",
            // SAFETY: - Bitmaps are trusted to contain a valid ptr (type invariant)
            //         - hwloc ops are trusted not to modify *const parameters
            || unsafe { hwlocality_sys::hwloc_bitmap_weight(self.as_ptr()) },
            -1,
        )
        .expect(SHOULD_NOT_FAIL);
        usize::try_from(result).ok()
    }

    /// Check the first unset index, if any
    ///
    /// You can iterate over set indices with [`Bitmap::iter_unset()`].
    ///
    /// # Examples
    ///
    /// ```
    /// use hwlocality::bitmap::Bitmap;
    ///
    /// let first_unset_usize = |b: Bitmap| b.first_unset().map(usize::from);
    /// assert_eq!(first_unset_usize(Bitmap::new()), Some(0));
    /// assert_eq!(first_unset_usize(Bitmap::from_range(..12)), Some(12));
    /// assert_eq!(Bitmap::full().first_unset(), None);
    /// ```
    #[doc(alias = "hwloc_bitmap_first_unset")]
    pub fn first_unset(&self) -> Option<BitmapIndex> {
        self.query_index(
            "hwloc_bitmap_first_unset",
            // SAFETY: - Bitmaps are trusted to contain a valid ptr (type invariant)
            //         - hwloc ops are trusted not to modify *const parameters
            || unsafe { hwlocality_sys::hwloc_bitmap_first_unset(self.as_ptr()) },
        )
    }

    /// Iterate over unset indices
    ///
    /// # Examples
    ///
    /// ```
    /// use hwlocality::bitmap::Bitmap;
    ///
    /// let bitmap = Bitmap::from_range(12..);
    /// let indices = bitmap.iter_unset().map(usize::from).collect::<Vec<_>>();
    /// assert_eq!(indices, &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
    /// ```
    #[doc(alias = "hwloc_bitmap_next_unset")]
    pub fn iter_unset(&self) -> Iter<&Self> {
        Iter::new(self, Self::next_unset)
    }

    /// Check the last unset index, if any
    ///
    /// # Examples
    ///
    /// ```
    /// use hwlocality::bitmap::Bitmap;
    ///
    /// let last_unset_usize = |b: Bitmap| b.last_unset().map(usize::from);
    /// assert_eq!(Bitmap::new().last_unset(), None);
    /// assert_eq!(last_unset_usize(Bitmap::from_range(12..)), Some(11));
    /// assert_eq!(Bitmap::full().last_unset(), None);
    /// ```
    #[doc(alias = "hwloc_bitmap_last_unset")]
    pub fn last_unset(&self) -> Option<BitmapIndex> {
        self.query_index(
            "hwloc_bitmap_last_unset",
            // SAFETY: - Bitmaps are trusted to contain a valid ptr (type invariant)
            //         - hwloc ops are trusted not to modify *const parameters
            || unsafe { hwlocality_sys::hwloc_bitmap_last_unset(self.as_ptr()) },
        )
    }

    /// Optimized version of `*self = !self`
    ///
    /// # Examples
    ///
    /// ```
    /// use hwlocality::bitmap::Bitmap;
    ///
    /// let mut bitmap = Bitmap::from_range(12..=34);
    /// bitmap.invert();
    /// assert_eq!(format!("{bitmap}"), "0-11,35-");
    /// ```
    pub fn invert(&mut self) {
        // SAFETY: - Bitmaps are trusted to contain a valid ptr (type invariant)
        //         - hwloc ops are trusted to keep *mut parameters in a
        //           valid state unless stated otherwise
        errors::call_hwloc_int_normal("hwloc_bitmap_not", || unsafe {
            hwlocality_sys::hwloc_bitmap_not(self.as_mut_ptr(), self.as_ptr())
        })
        .expect(MALLOC_FAIL_ONLY);
    }

    /// Truth that `self` and `rhs` have some set indices in common
    ///
    /// Accepts both `&'_ Bitmap` and `BitmapRef<'_, Bitmap>` operands.
    ///
    /// # Examples
    ///
    /// ```
    /// use hwlocality::bitmap::Bitmap;
    ///
    /// let bitmap1 = Bitmap::from_range(12..=34);
    /// let bitmap2 = Bitmap::from_range(56..=78);
    /// assert!(!bitmap1.intersects(&bitmap2));
    ///
    /// let bitmap3 = Bitmap::from_range(34..=56);
    /// assert!(bitmap1.intersects(&bitmap3));
    /// assert!(bitmap2.intersects(&bitmap3));
    /// ```
    #[doc(alias = "hwloc_bitmap_intersects")]
    pub fn intersects(&self, rhs: impl Deref<Target = Self>) -> bool {
        /// Polymorphized version of this function (avoids generics code bloat)
        fn polymorphized(self_: &Bitmap, rhs: &Bitmap) -> bool {
            // SAFETY: - Bitmaps are trusted to contain a valid ptr (type invariant)
            //         - hwloc ops are trusted not to modify *const parameters
            errors::call_hwloc_bool("hwloc_bitmap_intersects", || unsafe {
                hwlocality_sys::hwloc_bitmap_intersects(self_.as_ptr(), rhs.as_ptr())
            })
            .expect(SHOULD_NOT_FAIL)
        }
        polymorphized(self, &rhs)
    }

    /// Truth that the indices set in `inner` are a subset of those set in `self`
    ///
    /// Accepts both `&'_ Bitmap` and `BitmapRef<'_, Bitmap>` operands.
    ///
    /// The empty bitmap is considered included in any other bitmap.
    ///
    /// # Examples
    ///
    /// ```
    /// use hwlocality::bitmap::Bitmap;
    ///
    /// let bitmap1 = Bitmap::from_range(12..=78);
    /// let bitmap2 = Bitmap::from_range(34..=56);
    /// assert!(bitmap1.includes(&bitmap2));
    /// assert!(!bitmap2.includes(&bitmap1));
    /// ```
    #[doc(alias = "hwloc_bitmap_isincluded")]
    pub fn includes(&self, inner: impl Deref<Target = Self>) -> bool {
        /// Polymorphized version of this function (avoids generics code bloat)
        fn polymorphized(self_: &Bitmap, inner: &Bitmap) -> bool {
            // SAFETY: - Bitmaps are trusted to contain a valid ptr (type invariant)
            //         - hwloc ops are trusted not to modify *const parameters
            errors::call_hwloc_bool("hwloc_bitmap_isincluded", || unsafe {
                hwlocality_sys::hwloc_bitmap_isincluded(inner.as_ptr(), self_.as_ptr())
            })
            .expect(SHOULD_NOT_FAIL)
        }
        polymorphized(self, &inner)
    }

    // NOTE: When adding new methods, remember to add them to impl_newtype_ops too

    // === Implementation details ===

    /// Convert a Rust range to an hwloc range
    ///
    /// # Panics
    ///
    /// If `range` goes beyond the implementation-defined maximum index (at
    /// least 2^15-1, usually 2^31-1).
    //
    // --- Implementation details ---
    //
    // # Safety
    //
    // This function must produce a valid hwloc index range or dies trying.
    //
    // - Left bound must be in range 0..=c_int::MAX
    // - Right bound must be in range -1..=c_int::MAX, where -1 indicates
    //   "up to" infinity and other right bounds are inclusive
    // - Right bound is allowed to be smaller than left bound, this is used to
    //   encode empty ranges.
    fn hwloc_range<Idx>(range: impl RangeBounds<Idx>) -> (c_uint, c_int)
    where
        Idx: Copy + TryInto<BitmapIndex>,
        <Idx as TryInto<BitmapIndex>>::Error: Debug,
    {
        /// Literaly translate Rust range bounds into hwloc bounds if possible
        ///
        /// By literal translation, we mean that the left bounds and right
        /// bounds in the hwloc version are identical to the Rust version modulo
        /// shifting of exclusive bounds back into inclusive bounds and
        /// different encoding of infinite left/right edges.
        ///
        /// The bounds should have been pre-converted to `Option<BitmapIndex>`,
        /// this way all central code avoids generics code bloat.
        ///
        /// # Panics
        ///
        /// Panics if the user-specified bounds are too high.
        ///
        /// # Errors
        ///
        /// Return None if the Rust bounds are fine but cannot be literally
        /// translated to hwloc bounds. If a literal translation is not
        /// possible, it means either the start bound is [`BitmapIndex::MAX`]
        /// exclusive or the end bound is [`BitmapIndex::MIN`] exclusive. In
        /// both cases, the range covers no indices and can be replaced by any
        /// other empty range on the caller side.
        fn translate_bounds_literal(
            start: Option<Bound<BitmapIndex>>,
            end: Option<Bound<BitmapIndex>>,
        ) -> Option<(c_uint, c_int)> {
            let start = start.expect("Range start is out of the accepted 0..=c_int::MAX range");
            let start = match start {
                Bound::Unbounded => BitmapIndex::MIN,
                Bound::Included(i) => i,
                Bound::Excluded(i) => i.checked_add_signed(1)?,
            };
            let end = end.expect("Range end is out of the accepted 0..=c_int::MAX range");
            let end = match end {
                Bound::Unbounded => -1,
                Bound::Included(i) => i.to_c_int(),
                Bound::Excluded(i) => i.checked_add_signed(-1)?.to_c_int(),
            };
            Some((start.to_c_uint(), end))
        }

        // Translate bounds, using (1, 0) as the canonical empty hwloc range
        let convert_idx = |idx: &Idx| (*idx).try_into().ok();
        let convert_bound = |bound: Bound<&Idx>| match bound {
            Bound::Unbounded => Some(Bound::Unbounded),
            Bound::Included(idx) => convert_idx(idx).map(Bound::Included),
            Bound::Excluded(idx) => convert_idx(idx).map(Bound::Excluded),
        };
        translate_bounds_literal(
            convert_bound(range.start_bound()),
            convert_bound(range.end_bound()),
        )
        .unwrap_or((1, 0))
    }

    /// Common logic for first/last/next set/unset index queries
    fn query_index(&self, api: &'static str, call: impl FnOnce() -> c_int) -> Option<BitmapIndex> {
        let result = errors::call_hwloc_int_raw(api, call, -1).expect(SHOULD_NOT_FAIL);
        BitmapIndex::try_from_c_int(result).ok()
    }

    /// Iterator building block
    ///
    /// # Safety
    ///
    /// `next_fn` must be an hwloc entry point that takes a bitmap and start
    /// index as a parameter, and returns a next index as an output, such that...
    ///
    /// - The bitmap `*const` parameter is not modified by the operation
    /// - Start index can be -1 to find the first index matching a certain
    ///   criterion, or in `0..=c_int::MAX` to find the next index matching this
    ///   criterion after the specified index
    /// - Return value is the next index matching the selected criterion, or -1
    ///   to indicate absence of such index (and thus end of iteration)
    unsafe fn next(
        &self,
        api: &'static str,
        next_fn: impl FnOnce(*const hwloc_bitmap_s, c_int) -> c_int,
        index: Option<BitmapIndex>,
    ) -> Option<BitmapIndex> {
        self.query_index(api, || {
            // SAFETY: - Bitmaps are trusted to contain a valid ptr (type invariant)
            //         - hwloc ops are trusted not to modify *const parameters
            next_fn(self.as_ptr(), index.map_or(-1, BitmapIndex::to_c_int))
        })
    }

    /// Set index iterator building block
    fn next_set(&self, index: Option<BitmapIndex>) -> Option<BitmapIndex> {
        // SAFETY: This function is an hwloc bitmap iteration function
        unsafe {
            self.next(
                "hwloc_bitmap_next",
                |bitmap, prev| hwlocality_sys::hwloc_bitmap_next(bitmap, prev),
                index,
            )
        }
    }

    /// Unset index iterator building block
    fn next_unset(&self, index: Option<BitmapIndex>) -> Option<BitmapIndex> {
        // SAFETY: This function is an hwloc bitmap iteration function
        unsafe {
            self.next(
                "hwloc_bitmap_next_unset",
                |bitmap, prev| hwlocality_sys::hwloc_bitmap_next_unset(bitmap, prev),
                index,
            )
        }
    }
}

/// Generic error message for `usize -> BitmapIndex` conversion errors
const BAD_INDEX: &str = "Bitmap index is out of the accepted 0..=c_int::MAX range";

/// Common error message for operations that should only fail in the event
/// of a memory allocation failure, which is a panic in Rust
const MALLOC_FAIL_ONLY: &str =
    "This operation should only fail on malloc failure, which is a panic in Rust";

/// Common error message for operations that shouldn't fail
const SHOULD_NOT_FAIL: &str = "This operation has no known failure mode";

#[cfg(any(test, feature = "proptest"))]
impl Arbitrary for Bitmap {
    type Parameters = ();
    type Strategy = prop::strategy::Map<
        (
            prop::strategy::TupleUnion<(
                prop::strategy::WA<Just<HashSet<BitmapIndex>>>,
                prop::strategy::WA<
                    prop::collection::HashSetStrategy<crate::strategies::BitmapIndexStrategy>,
                >,
            )>,
            prop::bool::Any,
        ),
        fn((HashSet<BitmapIndex>, bool)) -> Self,
    >;

    fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
        use crate::strategies::bitmap_index;
        use prop::collection::SizeRange;
        let index_set = prop_oneof![
            // Bias towards generating more empty/full bitmaps, there are an
            // edge case of many algorithms
            1 => Just(HashSet::new()),
            4 => prop::collection::hash_set(bitmap_index(), SizeRange::default()),
        ];
        (index_set, prop::bool::ANY).prop_map(|(set_indices, invert)| {
            // Start with an arbitrary finite bitmap
            let mut result = set_indices.into_iter().collect::<Self>();

            // Decide by coin flip to invert it (which will make it infinite)
            if invert {
                result.invert();
            }

            result
        })
    }
}

impl<B: Borrow<Bitmap>> BitAnd<B> for &Bitmap {
    type Output = Bitmap;

    #[doc(alias = "hwloc_bitmap_and")]
    fn bitand(self, rhs: B) -> Bitmap {
        /// Polymorphized version of this function (avoids generics code bloat)
        fn polymorphized(self_: &Bitmap, rhs: &Bitmap) -> Bitmap {
            let mut result = Bitmap::new();
            // SAFETY: - Bitmaps are trusted to contain a valid ptr (type invariant)
            //         - hwloc ops are trusted not to modify *const parameters
            //         - hwloc ops are trusted to keep *mut parameters in a
            //           valid state unless stated otherwise
            errors::call_hwloc_int_normal("hwloc_bitmap_and", || unsafe {
                hwlocality_sys::hwloc_bitmap_and(result.as_mut_ptr(), self_.as_ptr(), rhs.as_ptr())
            })
            .expect(MALLOC_FAIL_ONLY);
            result
        }
        polymorphized(self, rhs.borrow())
    }
}

impl<B: Borrow<Self>> BitAnd<B> for Bitmap {
    type Output = Self;

    fn bitand(mut self, rhs: B) -> Self {
        self &= rhs.borrow();
        self
    }
}

impl<B: Borrow<Self>> BitAndAssign<B> for Bitmap {
    fn bitand_assign(&mut self, rhs: B) {
        /// Polymorphized version of this function (avoids generics code bloat)
        fn polymorphized(self_: &mut Bitmap, rhs: &Bitmap) {
            // SAFETY: - Bitmaps are trusted to contain a valid ptr (type invariant)
            //         - hwloc ops are trusted not to modify *const parameters
            //         - hwloc ops are trusted to keep *mut parameters in a
            //           valid state unless stated otherwise
            errors::call_hwloc_int_normal("hwloc_bitmap_and", || unsafe {
                hwlocality_sys::hwloc_bitmap_and(self_.as_mut_ptr(), self_.as_ptr(), rhs.as_ptr())
            })
            .expect(MALLOC_FAIL_ONLY);
        }
        polymorphized(self, rhs.borrow());
    }
}

impl<B: Borrow<Bitmap>> BitOr<B> for &Bitmap {
    type Output = Bitmap;

    #[doc(alias = "hwloc_bitmap_or")]
    fn bitor(self, rhs: B) -> Bitmap {
        /// Polymorphized version of this function (avoids generics code bloat)
        fn polymorphized(self_: &Bitmap, rhs: &Bitmap) -> Bitmap {
            let mut result = Bitmap::new();
            // SAFETY: - Bitmaps are trusted to contain a valid ptr (type invariant)
            //         - hwloc ops are trusted not to modify *const parameters
            //         - hwloc ops are trusted to keep *mut parameters in a
            //           valid state unless stated otherwise
            errors::call_hwloc_int_normal("hwloc_bitmap_or", || unsafe {
                hwlocality_sys::hwloc_bitmap_or(result.as_mut_ptr(), self_.as_ptr(), rhs.as_ptr())
            })
            .expect(MALLOC_FAIL_ONLY);
            result
        }
        polymorphized(self, rhs.borrow())
    }
}

impl<B: Borrow<Self>> BitOr<B> for Bitmap {
    type Output = Self;

    fn bitor(mut self, rhs: B) -> Self {
        self |= rhs.borrow();
        self
    }
}

impl<B: Borrow<Self>> BitOrAssign<B> for Bitmap {
    fn bitor_assign(&mut self, rhs: B) {
        /// Polymorphized version of this function (avoids generics code bloat)
        fn polymorphized(self_: &mut Bitmap, rhs: &Bitmap) {
            // SAFETY: - Bitmaps are trusted to contain a valid ptr (type invariant)
            //         - hwloc ops are trusted not to modify *const parameters
            //         - hwloc ops are trusted to keep *mut parameters in a
            //           valid state unless stated otherwise
            errors::call_hwloc_int_normal("hwloc_bitmap_or", || unsafe {
                hwlocality_sys::hwloc_bitmap_or(self_.as_mut_ptr(), self_.as_ptr(), rhs.as_ptr())
            })
            .expect(MALLOC_FAIL_ONLY);
        }
        polymorphized(self, rhs.borrow());
    }
}

impl<B: Borrow<Bitmap>> BitXor<B> for &Bitmap {
    type Output = Bitmap;

    #[doc(alias = "hwloc_bitmap_xor")]
    fn bitxor(self, rhs: B) -> Bitmap {
        /// Polymorphized version of this function (avoids generics code bloat)
        fn polymorphized(self_: &Bitmap, rhs: &Bitmap) -> Bitmap {
            let mut result = Bitmap::new();
            // SAFETY: - Bitmaps are trusted to contain a valid ptr (type invariant)
            //         - hwloc ops are trusted not to modify *const parameters
            //         - hwloc ops are trusted to keep *mut parameters in a
            //           valid state unless stated otherwise
            errors::call_hwloc_int_normal("hwloc_bitmap_xor", || unsafe {
                hwlocality_sys::hwloc_bitmap_xor(result.as_mut_ptr(), self_.as_ptr(), rhs.as_ptr())
            })
            .expect(MALLOC_FAIL_ONLY);
            result
        }
        polymorphized(self, rhs.borrow())
    }
}

impl<B: Borrow<Self>> BitXor<B> for Bitmap {
    type Output = Self;

    fn bitxor(mut self, rhs: B) -> Self {
        self ^= rhs.borrow();
        self
    }
}

impl<B: Borrow<Self>> BitXorAssign<B> for Bitmap {
    fn bitxor_assign(&mut self, rhs: B) {
        /// Polymorphized version of this function (avoids generics code bloat)
        fn polymorphized(self_: &mut Bitmap, rhs: &Bitmap) {
            // SAFETY: - Bitmaps are trusted to contain a valid ptr (type invariant)
            //         - hwloc ops are trusted not to modify *const parameters
            //         - hwloc ops are trusted to keep *mut parameters in a
            //           valid state unless stated otherwise
            errors::call_hwloc_int_normal("hwloc_bitmap_xor", || unsafe {
                hwlocality_sys::hwloc_bitmap_xor(self_.as_mut_ptr(), self_.as_ptr(), rhs.as_ptr())
            })
            .expect(MALLOC_FAIL_ONLY);
        }
        polymorphized(self, rhs.borrow());
    }
}

impl Clone for Bitmap {
    #[doc(alias = "hwloc_bitmap_dup")]
    fn clone(&self) -> Self {
        // SAFETY: - Bitmaps are trusted to contain a valid ptr (type invariant)
        //         - hwloc ops are trusted not to modify *const parameters
        //         - This operation is trusted to return a valid owned bitmap
        unsafe {
            let ptr = errors::call_hwloc_ptr_mut("hwloc_bitmap_dup", || {
                hwlocality_sys::hwloc_bitmap_dup(self.as_ptr())
            })
            .expect(MALLOC_FAIL_ONLY);
            Self::from_owned_nonnull(ptr)
        }
    }
}

impl Debug for Bitmap {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        <Self as Display>::fmt(self, f)
    }
}

impl Default for Bitmap {
    fn default() -> Self {
        Self::new()
    }
}

impl Display for Bitmap {
    #[doc(alias = "hwloc_bitmap_list_snprintf")]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        // SAFETY: - Bitmaps are trusted to contain a valid ptr (type invariant)
        //         - hwloc ops are trusted not to modify *const parameters
        //         - hwloc_bitmap_list_snprintf is snprintf-like
        unsafe {
            ffi::write_snprintf(f, |buf, len| {
                hwlocality_sys::hwloc_bitmap_list_snprintf(buf, len, self.as_ptr())
            })
        }
    }
}

impl Drop for Bitmap {
    #[doc(alias = "hwloc_bitmap_free")]
    fn drop(&mut self) {
        // SAFETY: - Bitmaps are trusted to contain a valid ptr (type invariant)
        //         - Only owned bitmaps should be exposed to the user in a
        //           droppable state like &mut self or owned Self
        //         - Bitmap will not be usable again after Drop
        unsafe { hwlocality_sys::hwloc_bitmap_free(self.as_mut_ptr()) }
    }
}

impl Eq for Bitmap {}

impl<BI: Borrow<BitmapIndex>> Extend<BI> for Bitmap {
    fn extend<T: IntoIterator<Item = BI>>(&mut self, iter: T) {
        for i in iter {
            self.set(*i.borrow());
        }
    }
}

impl<BI: Borrow<BitmapIndex>> From<BI> for Bitmap {
    fn from(value: BI) -> Self {
        let mut result = Self::new();
        result.set(*value.borrow());
        result
    }
}

impl<BI: Borrow<BitmapIndex>> FromIterator<BI> for Bitmap {
    fn from_iter<I: IntoIterator<Item = BI>>(iter: I) -> Self {
        let mut bitmap = Self::new();
        bitmap.extend(iter);
        bitmap
    }
}

impl Hash for Bitmap {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        // Beware not to iterate infinitely over infinite bitmaps
        let is_infinitely_set = self.weight().is_none();
        let last_significant_index = self
            .last_set()
            .or_else(|| self.last_unset())
            .unwrap_or(BitmapIndex::MIN);

        // Iterate over the finite part of the bitmap
        for index in self
            .iter_set()
            .take_while(|&idx| idx <= last_significant_index)
        {
            index.hash(state);
        }

        // Add a terminator that is different depending on if what follows the
        // finite part of the bitmap is an infinite sequence of 1s or 0s.
        is_infinitely_set.hash(state);
    }
}

/// Iterator over set or unset [`Bitmap`] indices
#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub struct Iter<B> {
    /// Bitmap over which we're iterating
    bitmap: B,

    /// Last explored index
    prev: Option<BitmapIndex>,

    /// Mapping from last index to next index
    next: fn(&Bitmap, Option<BitmapIndex>) -> Option<BitmapIndex>,
}
//
impl<B> Iter<B> {
    /// Set up a bitmap iterator
    fn new(bitmap: B, next: fn(&Bitmap, Option<BitmapIndex>) -> Option<BitmapIndex>) -> Self {
        Self {
            bitmap,
            prev: None,
            next,
        }
    }
}
//
impl<B: Borrow<Bitmap>> Iterator for Iter<B> {
    type Item = BitmapIndex;

    fn next(&mut self) -> Option<BitmapIndex> {
        self.prev = (self.next)(self.bitmap.borrow(), self.prev);
        self.prev
    }
}
//
impl<B: Borrow<Bitmap>> FusedIterator for Iter<B> {}
//
impl<'bitmap> IntoIterator for &'bitmap Bitmap {
    type Item = BitmapIndex;
    type IntoIter = Iter<Self>;

    fn into_iter(self) -> Self::IntoIter {
        Iter::new(self, Bitmap::next_set)
    }
}
//
impl IntoIterator for Bitmap {
    type Item = BitmapIndex;
    type IntoIter = Iter<Self>;

    fn into_iter(self) -> Self::IntoIter {
        Iter::new(self, Self::next_set)
    }
}

impl Not for &Bitmap {
    type Output = Bitmap;

    #[doc(alias = "hwloc_bitmap_not")]
    fn not(self) -> Bitmap {
        let mut result = Bitmap::new();
        // SAFETY: - Bitmaps are trusted to contain a valid ptr (type invariant)
        //         - hwloc ops are trusted not to modify *const parameters
        //         - hwloc ops are trusted to keep *mut parameters in a
        //           valid state unless stated otherwise
        errors::call_hwloc_int_normal("hwloc_bitmap_not", || unsafe {
            hwlocality_sys::hwloc_bitmap_not(result.as_mut_ptr(), self.as_ptr())
        })
        .expect(MALLOC_FAIL_ONLY);
        result
    }
}

impl Not for Bitmap {
    type Output = Self;

    fn not(mut self) -> Self {
        self.invert();
        self
    }
}

impl Ord for Bitmap {
    #[doc(alias = "hwloc_bitmap_compare")]
    fn cmp(&self, other: &Self) -> Ordering {
        // SAFETY: - Bitmaps are trusted to contain a valid ptr (type invariant)
        //         - hwloc ops are trusted not to modify *const parameters
        let result = unsafe { hwlocality_sys::hwloc_bitmap_compare(self.as_ptr(), other.as_ptr()) };
        match result {
            -1 => Ordering::Less,
            0 => Ordering::Equal,
            1 => Ordering::Greater,
            _ => unreachable!("hwloc_bitmap_compare returned unexpected result {result}"),
        }
    }
}

impl<B: Borrow<Self>> PartialEq<B> for Bitmap {
    #[doc(alias = "hwloc_bitmap_isequal")]
    fn eq(&self, other: &B) -> bool {
        /// Polymorphized version of this function (avoids generics code bloat)
        fn polymorphized(self_: &Bitmap, other: &Bitmap) -> bool {
            // SAFETY: - Bitmaps are trusted to contain a valid ptr (type invariant)
            //         - hwloc ops are trusted not to modify *const parameters
            errors::call_hwloc_bool("hwloc_bitmap_isequal", || unsafe {
                hwlocality_sys::hwloc_bitmap_isequal(self_.as_ptr(), other.as_ptr())
            })
            .expect(SHOULD_NOT_FAIL)
        }
        polymorphized(self, other.borrow())
    }
}

impl<B: Borrow<Self>> PartialOrd<B> for Bitmap {
    fn partial_cmp(&self, other: &B) -> Option<Ordering> {
        Some(self.cmp(other.borrow()))
    }
}

impl Pointer for Bitmap {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        <NonNull<hwloc_bitmap_s> as fmt::Pointer>::fmt(&self.0, f)
    }
}

// SAFETY: Safe because Bitmap exposes no internal mutability
unsafe impl Send for Bitmap {}

impl<B: Borrow<Bitmap>> Sub<B> for &Bitmap {
    type Output = Bitmap;

    #[doc(alias = "hwloc_bitmap_andnot")]
    fn sub(self, rhs: B) -> Bitmap {
        /// Polymorphized version of this function (avoids generics code bloat)
        fn polymorphized(self_: &Bitmap, rhs: &Bitmap) -> Bitmap {
            let mut result = Bitmap::new();
            // SAFETY: - Bitmaps are trusted to contain a valid ptr (type invariant)
            //         - hwloc ops are trusted not to modify *const parameters
            //         - hwloc ops are trusted to keep *mut parameters in a
            //           valid state unless stated otherwise
            errors::call_hwloc_int_normal("hwloc_bitmap_andnot", || unsafe {
                hwlocality_sys::hwloc_bitmap_andnot(
                    result.as_mut_ptr(),
                    self_.as_ptr(),
                    rhs.as_ptr(),
                )
            })
            .expect(MALLOC_FAIL_ONLY);
            result
        }
        polymorphized(self, rhs.borrow())
    }
}

impl<B: Borrow<Self>> Sub<B> for Bitmap {
    type Output = Self;

    fn sub(mut self, rhs: B) -> Self {
        self -= rhs.borrow();
        self
    }
}

impl<B: Borrow<Self>> SubAssign<B> for Bitmap {
    fn sub_assign(&mut self, rhs: B) {
        /// Polymorphized version of this function (avoids generics code bloat)
        fn polymorphized(self_: &mut Bitmap, rhs: &Bitmap) {
            // SAFETY: - Bitmaps are trusted to contain a valid ptr (type invariant)
            //         - hwloc ops are trusted not to modify *const parameters
            //         - hwloc ops are trusted to keep *mut parameters in a
            //           valid state unless stated otherwise
            errors::call_hwloc_int_normal("hwloc_bitmap_andnot", || unsafe {
                hwlocality_sys::hwloc_bitmap_andnot(
                    self_.as_mut_ptr(),
                    self_.as_ptr(),
                    rhs.as_ptr(),
                )
            })
            .expect(MALLOC_FAIL_ONLY);
        }
        polymorphized(self, rhs.borrow());
    }
}

// SAFETY: Safe because Bitmap exposes no internal mutability
unsafe impl Sync for Bitmap {}

#[allow(clippy::cognitive_complexity, clippy::op_ref, clippy::too_many_lines)]
#[cfg(test)]
pub(crate) mod tests {
    use super::reference::tests::{test_bitmap_ref_binops, test_bitmap_ref_unary};
    use super::*;
    use crate::strategies::bitmap_index;
    use proptest::sample::SizeRange;
    #[allow(unused)]
    use similar_asserts::assert_eq;
    use static_assertions::{
        assert_eq_align, assert_eq_size, assert_impl_all, assert_not_impl_any, assert_type_eq_all,
    };
    use std::{
        collections::hash_map::RandomState,
        error::Error,
        ffi::c_ulonglong,
        fmt::{Binary, LowerExp, LowerHex, Octal, UpperExp, UpperHex, Write},
        hash::BuildHasher,
        io::{self, Read},
        mem::ManuallyDrop,
        ops::{RangeFrom, RangeInclusive},
        panic::UnwindSafe,
        ptr,
    };

    // Check that public types in this module keep implementing all expected
    // traits, in the interest of detecting future semver-breaking changes
    assert_eq_align!(Bitmap, NonNull<hwloc_bitmap_s>);
    assert_eq_size!(Bitmap, NonNull<hwloc_bitmap_s>);
    assert_impl_all!(Bitmap:
        BitAnd<Bitmap>, BitAnd<&'static Bitmap>,
        BitAndAssign<Bitmap>, BitAndAssign<&'static Bitmap>,
        BitOr<Bitmap>, BitOr<&'static Bitmap>,
        BitOrAssign<Bitmap>, BitOrAssign<&'static Bitmap>,
        BitXor<Bitmap>, BitXor<&'static Bitmap>,
        BitXorAssign<Bitmap>, BitXorAssign<&'static Bitmap>,
        Clone, Debug, Default, Display,
        Extend<BitmapIndex>, Extend<&'static BitmapIndex>,
        From<BitmapIndex>, From<&'static BitmapIndex>,
        FromIterator<BitmapIndex>, FromIterator<&'static BitmapIndex>,
        Hash, IntoIterator<Item=BitmapIndex>, Not, Ord, OwnedBitmap,
        PartialEq<&'static Bitmap>,
        PartialOrd<&'static Bitmap>,
        Pointer, Sized,
        Sub<Bitmap>, Sub<&'static Bitmap>,
        SubAssign<Bitmap>, SubAssign<&'static Bitmap>,
        Sync, Unpin, UnwindSafe
    );
    assert_not_impl_any!(Bitmap:
        Binary, Copy, Deref, Error, LowerExp, LowerHex, Octal, Read, UpperExp,
        UpperHex, fmt::Write, io::Write
    );
    assert_impl_all!(&Bitmap:
        BitAnd<Bitmap>, BitAnd<&'static Bitmap>,
        BitOr<Bitmap>, BitOr<&'static Bitmap>,
        BitXor<Bitmap>, BitXor<&'static Bitmap>,
        IntoIterator<Item=BitmapIndex>,
        Not<Output=Bitmap>,
        Sub<Bitmap>, Sub<&'static Bitmap>,
    );
    assert_type_eq_all!(BitmapIndex, PositiveInt);
    assert_impl_all!(Iter<Bitmap>:
        Clone, Debug, FusedIterator<Item=BitmapIndex>, Hash, Sized, Sync, Unpin,
        UnwindSafe
    );
    assert_not_impl_any!(Iter<Bitmap>:
        Binary, Copy, Default, Deref, Display, LowerExp, LowerHex, Octal,
        Pointer, Read, UpperExp, UpperHex, fmt::Write, io::Write
    );
    assert_impl_all!(Iter<&Bitmap>:
        Copy, Debug, FusedIterator<Item=BitmapIndex>, Hash, Sized, Sync, Unpin,
        UnwindSafe
    );
    assert_not_impl_any!(Iter<&Bitmap>:
        Binary, Default, Deref, Display, LowerExp, LowerHex, Octal, Pointer,
        Read, UpperExp, UpperHex, fmt::Write, io::Write
    );

    // We can't fully check the value of infinite iterators because that would
    // literally take forever, so we only check a small subrange of the final
    // all-set/unset region, large enough to catch off-by-one-longword issues.
    pub(crate) const INFINITE_EXPLORE_ITERS: usize = std::mem::size_of::<c_ulonglong>() * 8;

    // Unfortunately, ranges of BitmapIndex cannot do everything that ranges of
    // built-in integer types can do due to some unstable integer traits, so
    // it's sometimes good to go back to usize.
    fn range_to_usize(range: &RangeInclusive<BitmapIndex>) -> RangeInclusive<usize> {
        usize::from(*range.start())..=usize::from(*range.end())
    }

    // Split a possibly infinite bitmap into a finite bitmap and an infinite
    // range of set indices, separated from the indices of the finite bitmap by
    // a range of unset indices. To get the original bitmap back, use `set_range`.
    fn split_infinite_bitmap(mut bitmap: Bitmap) -> (Bitmap, Option<RangeFrom<BitmapIndex>>) {
        // If this bitmap is infinite...
        if bitmap.weight().is_none() {
            // ...and it has a finite part...
            bitmap.last_unset().map_or_else(
                || (Bitmap::new(), Some(BitmapIndex::MIN..)),
                |last_unset| {
                    let infinite_part = last_unset.checked_add_signed(1).unwrap()..;
                    bitmap.unset_range(infinite_part.clone());
                    (bitmap, Some(infinite_part))
                },
            )
        } else {
            (bitmap, None)
        }
    }

    fn test_basic_inplace(initial: &Bitmap, inverse: &Bitmap) -> Result<(), TestCaseError> {
        prop_assert_eq!(format!("{:p}", *initial), format!("{:p}", initial.0));

        let mut buf = initial.clone();
        buf.clear();
        prop_assert!(buf.is_empty());

        buf.copy_from(initial);
        buf.fill();
        prop_assert!(buf.is_full());

        buf.copy_from(initial);
        buf.invert();
        prop_assert_eq!(&buf, inverse);

        if initial.weight().unwrap_or(usize::MAX) > 0 {
            buf.copy_from(initial);
            buf.singlify();
            prop_assert_eq!(buf.weight(), Some(1));
        }
        Ok(())
    }

    fn test_indexing(
        initial: &Bitmap,
        index: BitmapIndex,
        initially_set: bool,
    ) -> Result<(), TestCaseError> {
        let single = Bitmap::from(index);
        let single_hole = !&single;

        // Bitmaps are conceptually infinite so we must be careful with
        // iteration-based verification of full bitmap contents. Let's just
        // account for off-by-one-word errors.
        let max_iters = initial
            .weight()
            .unwrap_or_else(|| usize::from(index) + INFINITE_EXPLORE_ITERS);

        prop_assert_eq!(initial.is_set(index), initially_set);

        let mut buf = initial.clone();
        buf.set(index);
        prop_assert_eq!(
            buf.weight(),
            initial.weight().map(|w| w + usize::from(!initially_set))
        );
        for idx in std::iter::once(index).chain(initial.iter_set().take(max_iters)) {
            prop_assert!(buf.is_set(idx));
        }

        buf.copy_from(initial);
        buf.set_only(index);
        prop_assert_eq!(&buf, &single);

        buf.copy_from(initial);
        buf.set_all_but(index);
        prop_assert_eq!(&buf, &single_hole);

        buf.copy_from(initial);
        buf.unset(index);
        prop_assert_eq!(
            buf.weight(),
            initial.weight().map(|w| w - usize::from(initially_set))
        );
        for idx in initial.iter_set().take(max_iters) {
            prop_assert_eq!(buf.is_set(idx), idx != index);
        }
        Ok(())
    }

    fn test_and_sub(b1: &Bitmap, b2: &Bitmap, and: &Bitmap) -> Result<(), TestCaseError> {
        prop_assert_eq!(&(b1 & b2), and);
        prop_assert_eq!(&(b1.clone() & b2), and);
        let mut buf = b1.clone();
        buf &= b2;
        prop_assert_eq!(&buf, and);

        let b1_andnot_b2 = b1 & !b2;
        prop_assert_eq!(&(b1 - b2), &b1_andnot_b2);
        prop_assert_eq!(&(b1.clone() - b2), &b1_andnot_b2);
        buf.copy_from(b1);
        buf -= b2;
        prop_assert_eq!(&buf, &b1_andnot_b2);

        let b2_andnot_b1 = b2 & !b1;
        prop_assert_eq!(&(b2 - b1), &b2_andnot_b1);
        prop_assert_eq!(&(b2.clone() - b1), &b2_andnot_b1);
        buf.copy_from(b2);
        buf -= b1;
        prop_assert_eq!(buf, b2_andnot_b1);
        Ok(())
    }

    #[test]
    fn test_low_level_null() {
        // SAFETY: All of the following are safe on null input
        unsafe {
            assert_eq!(Bitmap::from_owned_raw_mut(ptr::null_mut()), None);
            assert_eq!(Bitmap::borrow_from_raw(ptr::null()), None);
            assert_eq!(Bitmap::borrow_from_raw_mut(ptr::null_mut()), None);
        }
    }

    fn test_low_level_nonnull(bitmap: Bitmap) -> Result<(), TestCaseError> {
        test_bitmap_ref_unary(&bitmap, BitmapRef::from(&bitmap))?;
        let bitmap = ManuallyDrop::new(bitmap);
        let inner = bitmap.0;
        // SAFETY: We won't use this pointer to mutate
        prop_assert_eq!(unsafe { bitmap.inner() }, inner);
        {
            // SAFETY: If it worked for the original bitmap, it works here too,
            //         as long as we leave the original aside and refrain from
            //         dropping the bitmap on either side.
            let bitmap = unsafe { Bitmap::from_owned_nonnull(inner) };
            prop_assert_eq!(bitmap.0, inner);
            std::mem::forget(bitmap);
        }
        {
            // SAFETY: If it worked for the original bitmap, it works here too,
            //         as long as we leave the original aside and refrain from
            //         dropping the bitmap on either side.
            let bitmap = unsafe { Bitmap::from_owned_raw_mut(inner.as_ptr()).unwrap() };
            prop_assert_eq!(bitmap.0, inner);
            std::mem::forget(bitmap);
        }
        let bitmap = ManuallyDrop::into_inner(bitmap);
        {
            // SAFETY: Safe as long as we don't invalidate the original
            let borrow = unsafe { Bitmap::borrow_from_nonnull(inner) };
            prop_assert_eq!(borrow.0, inner);
            test_bitmap_ref_unary(&bitmap, borrow)?;
        }
        {
            // SAFETY: Safe as long as we don't invalidate the original
            let borrow = unsafe { Bitmap::borrow_from_raw(bitmap.as_ptr()).unwrap() };
            prop_assert_eq!(borrow.0, inner);
            test_bitmap_ref_unary(&bitmap, borrow)?;
        }
        let mut bitmap = bitmap;
        {
            // SAFETY: Safe as long as we don't invalidate the original
            let borrow = unsafe { Bitmap::borrow_from_raw_mut(bitmap.as_mut_ptr()).unwrap() };
            prop_assert_eq!(borrow.0, inner);
            test_bitmap_ref_unary(&bitmap, borrow)?;
        }
        Ok(())
    }

    #[allow(clippy::redundant_clone)]
    #[test]
    fn empty() -> Result<(), TestCaseError> {
        let empty = Bitmap::new();
        let mut empty2 = Bitmap::full();
        empty2.unset_range::<BitmapIndex>(..);
        let inverse = Bitmap::full();

        let test_empty = |empty: &Bitmap| {
            prop_assert_eq!(empty.first_set(), None);
            prop_assert_eq!(empty.first_unset().map(usize::from), Some(0));
            prop_assert!(empty.is_empty());
            prop_assert!(!empty.is_full());
            prop_assert_eq!(empty.into_iter().count(), 0);
            prop_assert_eq!(empty.iter_set().count(), 0);
            prop_assert_eq!(empty.last_set(), None);
            prop_assert_eq!(empty.last_unset(), None);
            prop_assert_eq!(empty.weight(), Some(0));

            for (expected, idx) in empty.iter_unset().enumerate().take(INFINITE_EXPLORE_ITERS) {
                prop_assert_eq!(expected, usize::from(idx));
            }
            for (expected, idx) in empty
                .clone()
                .into_iter()
                .enumerate()
                .take(INFINITE_EXPLORE_ITERS)
            {
                prop_assert_eq!(expected, usize::from(idx));
            }

            prop_assert_eq!(format!("{empty:?}"), "");
            prop_assert_eq!(format!("{empty}"), "");
            prop_assert_eq!(&(!empty), &inverse);
            prop_assert_eq!(&(!(empty.clone())), &inverse);
            Ok(())
        };
        test_empty(&empty)?;
        test_empty(&empty.clone())?;
        test_empty(&empty2)?;
        test_empty(&Bitmap::default())?;

        test_basic_inplace(&empty, &inverse)?;

        test_low_level_nonnull(empty)?;
        Ok(())
    }

    /// Generate a ranges of contiguous bitmap indices
    pub(crate) fn index_range() -> impl Strategy<Value = RangeInclusive<BitmapIndex>> {
        [bitmap_index(), bitmap_index()].prop_map(|[start, end]| start..=end)
    }

    /// Generate a set of possibly duplicate bitmap indices
    pub(crate) fn index_vec() -> impl Strategy<Value = Vec<BitmapIndex>> {
        prop::collection::vec(bitmap_index(), SizeRange::default())
    }

    /// Generate a vector of indices and the associated deduplicated set
    fn index_vec_and_set() -> impl Strategy<Value = (Vec<BitmapIndex>, HashSet<BitmapIndex>)> {
        index_vec().prop_map(|v| {
            let s = v.iter().copied().collect::<HashSet<_>>();
            (v, s)
        })
    }

    proptest! {
        #[test]
        fn empty_extend((extra, extra_set) in index_vec_and_set()) {
            let mut extended = Bitmap::new();
            extended.extend(extra);

            prop_assert_eq!(extended.weight(), Some(extra_set.len()));
            for idx in extra_set {
                prop_assert!(extended.is_set(idx));
            }
        }

        #[test]
        fn empty_op_index(index in bitmap_index()) {
            test_indexing(&Bitmap::new(), index, false)?;
        }

        #[test]
        fn empty_op_range(range in index_range()) {
            let mut buf = Bitmap::new();
            buf.set_range(range.clone());
            prop_assert_eq!(&buf, &Bitmap::from_range(range.clone()));
            buf.clear();

            buf.unset_range(range);
            prop_assert!(buf.is_empty());
        }

        #[test]
        fn empty_op_bitmap(other: Bitmap) {
            let empty = Bitmap::new();

            prop_assert_eq!(empty.includes(&other), other.is_empty());
            prop_assert!(other.includes(&empty));
            prop_assert!(!empty.intersects(&other));

            prop_assert_eq!(empty == other, other.is_empty());
            if other.is_empty() {
                let state = RandomState::new();
                prop_assert_eq!(state.hash_one(&other), state.hash_one(&empty));
            } else {
                prop_assert!(empty < other);
            }

            test_and_sub(&empty, &other, &empty)?;

            prop_assert_eq!(&(&empty | &other), &other);
            prop_assert_eq!(&(empty.clone() | &other), &other);
            let mut buf = Bitmap::new();
            buf |= &other;
            prop_assert_eq!(&buf, &other);

            prop_assert_eq!(&(&empty ^ &other), &other);
            prop_assert_eq!(&(empty.clone() ^ &other), &other);
            buf.clear();
            buf ^= &other;
            prop_assert_eq!(&buf, &other);

            test_bitmap_ref_binops(&empty, &other)?;
        }
    }

    #[allow(clippy::redundant_clone)]
    #[test]
    fn full() -> Result<(), TestCaseError> {
        let full = Bitmap::full();
        let full2 = Bitmap::from_range::<BitmapIndex>(..);
        let mut full3 = Bitmap::new();
        full3.set_range::<BitmapIndex>(..);
        let inverse = Bitmap::new();

        let test_full = |full: &Bitmap| {
            prop_assert_eq!(full.first_set().map(usize::from), Some(0));
            prop_assert_eq!(full.first_unset(), None);
            prop_assert!(!full.is_empty());
            prop_assert!(full.is_full());
            prop_assert_eq!(full.iter_unset().count(), 0);
            prop_assert_eq!(full.last_set(), None);
            prop_assert_eq!(full.last_unset(), None);
            prop_assert_eq!(full.weight(), None);

            fn test_iter_set(iter: impl Iterator<Item = BitmapIndex>) -> Result<(), TestCaseError> {
                for (expected, idx) in iter.enumerate().take(INFINITE_EXPLORE_ITERS) {
                    prop_assert_eq!(expected, usize::from(idx));
                }
                Ok(())
            }
            test_iter_set(full.into_iter())?;
            test_iter_set(full.clone().into_iter())?;
            test_iter_set(full.iter_set())?;

            prop_assert_eq!(format!("{full:?}"), "0-");
            prop_assert_eq!(format!("{full}"), "0-");
            prop_assert_eq!(&(!full), &inverse);
            prop_assert_eq!(&(!(full.clone())), &inverse);
            Ok(())
        };
        test_full(&full)?;
        test_full(&full.clone())?;
        test_full(&full2)?;
        test_full(&full3)?;

        test_basic_inplace(&full, &inverse)?;
        test_low_level_nonnull(full)?;
        Ok(())
    }

    proptest! {
        #[test]
        fn full_extend(extra in index_vec()) {
            let mut extended = Bitmap::full();
            extended.extend(extra.iter().copied());
            prop_assert!(extended.is_full());
        }

        #[test]
        fn full_op_index(index in bitmap_index()) {
            test_indexing(&Bitmap::full(), index, true)?;
        }

        #[test]
        fn full_op_range(range in index_range()) {
            let mut ranged_hole = Bitmap::from_range(range.clone());
            ranged_hole.invert();

            let mut buf = Bitmap::full();
            buf.set_range(range.clone());
            prop_assert!(buf.is_full());

            buf.fill();
            buf.unset_range(range);
            prop_assert_eq!(buf, ranged_hole);
        }

        #[test]
        fn full_op_bitmap(other: Bitmap) {
            let full = Bitmap::full();
            let not_other = !&other;

            prop_assert!(full.includes(&other));
            prop_assert_eq!(other.includes(&full), other.is_full());
            prop_assert_eq!(full.intersects(&other), !other.is_empty());

            prop_assert_eq!(full == other, other.is_full());
            if other.is_full() {
                let state = RandomState::new();
                prop_assert_eq!(state.hash_one(&other), state.hash_one(&full));
            }
            prop_assert_eq!(
                full.cmp(&other),
                if other.is_full() {
                    Ordering::Equal
                } else {
                    Ordering::Greater
                }
            );

            test_and_sub(&full, &other, &other)?;

            prop_assert!((&full | &other).is_full());
            prop_assert!((full.clone() | &other).is_full());
            let mut buf = Bitmap::full();
            buf |= &other;
            prop_assert!(buf.is_full());

            prop_assert_eq!(&(&full ^ &other), &not_other);
            prop_assert_eq!(&(full.clone() ^ &other), &not_other);
            buf.fill();
            buf ^= &other;
            prop_assert_eq!(buf, not_other);

            test_bitmap_ref_binops(&full, &other)?;
        }

        #[allow(clippy::redundant_clone)]
        #[test]
        fn from_range(range in index_range()) {
            let ranged_bitmap = Bitmap::from_range(range.clone());

            // Predict bitmap properties from range properties
            let elems = range_to_usize(&range)
                .map(|idx| BitmapIndex::try_from(idx).unwrap())
                .collect::<Vec<_>>();
            let first_unset = if elems.first() == Some(&BitmapIndex::MIN) {
                elems
                    .last()
                    .copied()
                    .and_then(|item| item.checked_add_signed(1))
            } else {
                Some(BitmapIndex::MIN)
            };
            let unset_after_set = elems.last().map_or(Some(BitmapIndex::MIN), |last_set| {
                last_set.checked_add_signed(1)
            });
            let display = if let (Some(first), Some(last)) = (elems.first(), elems.last()) {
                if first == last {
                    format!("{first}")
                } else {
                    format!("{first}-{last}")
                }
            } else {
                String::new()
            };
            let inverse = if let (Some(&first), Some(last)) = (elems.first(), elems.last()) {
                let mut buf = Bitmap::from_range(..first);
                if let Some(after_last) = last.checked_add_signed(1) {
                    buf.set_range(after_last..)
                }
                buf
            } else {
                Bitmap::full()
            };

            // Check that the bitmap has the expected properties
            let test_ranged = |ranged_bitmap: &Bitmap| {
                prop_assert_eq!(ranged_bitmap.first_set(), elems.first().copied());
                prop_assert_eq!(ranged_bitmap.first_unset(), first_unset);
                prop_assert_eq!(ranged_bitmap.is_empty(), elems.is_empty());
                prop_assert!(!ranged_bitmap.is_full());
                prop_assert_eq!(&ranged_bitmap.into_iter().collect::<Vec<_>>(), &elems);
                prop_assert_eq!(&ranged_bitmap.clone().into_iter().collect::<Vec<_>>(), &elems);
                prop_assert_eq!(&ranged_bitmap.iter_set().collect::<Vec<_>>(), &elems);
                prop_assert_eq!(&ranged_bitmap.last_set(), &elems.last().copied());
                prop_assert_eq!(ranged_bitmap.last_unset(), None);
                prop_assert_eq!(ranged_bitmap.weight(), Some(elems.len()));

                let mut unset = ranged_bitmap.iter_unset();
                if let Some(first_set) = elems.first() {
                    for expected_unset in 0..usize::from(*first_set) {
                        prop_assert_eq!(unset.next().map(usize::from), Some(expected_unset));
                    }
                }
                let mut expected_unset =
                    std::iter::successors(unset_after_set, |unset| unset.checked_add_signed(1));
                for unset_index in unset.take(INFINITE_EXPLORE_ITERS) {
                    prop_assert_eq!(unset_index, expected_unset.next().unwrap())
                }

                prop_assert_eq!(&format!("{ranged_bitmap:?}"), &display);
                prop_assert_eq!(&format!("{ranged_bitmap}"), &display);
                prop_assert_eq!(&(!ranged_bitmap), &inverse);
                prop_assert_eq!(&(!(ranged_bitmap.clone())), &inverse);
                Ok(())
            };
            test_ranged(&ranged_bitmap)?;
            test_ranged(&ranged_bitmap.clone())?;

            // Run unary tests common to all bitmaps
            test_basic_inplace(&ranged_bitmap, &inverse)?;
            test_low_level_nonnull(ranged_bitmap.clone())?;

            // Quickly check other kinds of ranges
            let mut exclude_left = Bitmap::from_range((
                Bound::Excluded(*range.start()),
                Bound::Included(*range.end()),
            ));
            prop_assert!(!exclude_left.is_set(*range.start()));
            if range.contains(range.start()) {
                exclude_left.set(*range.start());
            }
            prop_assert_eq!(&exclude_left, &ranged_bitmap);
            //
            let mut exclude_right = Bitmap::from_range(*range.start()..*range.end());
            prop_assert!(!exclude_right.is_set(*range.end()));
            if range.contains(range.end()) {
                exclude_right.set(*range.end());
            }
            prop_assert_eq!(exclude_right, ranged_bitmap);
        }

        #[test]
        fn from_range_extend(
            range in index_range(),
            (extra, extra_set) in index_vec_and_set()
        ) {
            let mut extended = Bitmap::from_range(range.clone());
            extended.extend(extra);

            let mut predicted = extra_set;
            for idx in usize::from(*range.start())..=usize::from(*range.end()) {
                predicted.insert(idx.try_into().unwrap());
            }

            prop_assert_eq!(extended.weight(), Some(predicted.len()));
            for idx in predicted {
                prop_assert!(extended.is_set(idx));
            }
        }

        #[test]
        fn from_range_op_index(range in index_range(), index in bitmap_index()) {
            test_indexing(
                &Bitmap::from_range(range.clone()),
                index,
                range.contains(&index),
            )?;
        }

        #[test]
        fn from_range_op_range(
            [range1, range2] in [index_range(), index_range()]
        ) {
            let usized = range_to_usize(&range1);
            let other_usized = range_to_usize(&range2);

            let num_indices = |range: &RangeInclusive<usize>| range.clone().count();
            let num_common_indices = if usized.is_empty() || other_usized.is_empty() {
                0
            } else {
                num_indices(
                    &(*usized.start().max(other_usized.start())
                        ..=*usized.end().min(other_usized.end())),
                )
            };

            let ranged_bitmap = Bitmap::from_range(range1);

            let mut buf = ranged_bitmap.clone();
            buf.set_range(range2.clone());
            prop_assert_eq!(
                buf.weight().unwrap(),
                num_indices(&usized) + num_indices(&other_usized) - num_common_indices
            );
            for idx in usized.clone().chain(other_usized.clone()) {
                prop_assert!(buf.is_set(idx));
            }

            buf.copy_from(&ranged_bitmap);
            buf.unset_range(range2);
            prop_assert_eq!(
                buf.weight().unwrap(),
                num_indices(&usized) - num_common_indices
            );
            for idx in usized {
                prop_assert_eq!(buf.is_set(idx), !other_usized.contains(&idx));
            }
        }

        #[allow(clippy::similar_names)]
        #[test]
        fn from_range_op_bitmap(
            range in index_range(),
            other: Bitmap
        ) {
            let ranged_bitmap = Bitmap::from_range(range.clone());
            let usized = range_to_usize(&range);

            prop_assert_eq!(
                ranged_bitmap.includes(&other),
                other.is_empty()
                    || (other.last_set().is_some() && other.iter_set().all(|idx| range.contains(&idx)))
            );
            prop_assert_eq!(
                other.includes(&ranged_bitmap),
                usized.clone().all(|idx| other.is_set(idx))
            );
            prop_assert_eq!(
                ranged_bitmap.intersects(&other),
                usized.clone().any(|idx| other.is_set(idx))
            );

            prop_assert_eq!(
                ranged_bitmap == other,
                other.weight() == Some(usized.count()) && other.includes(&ranged_bitmap)
            );
            if ranged_bitmap == other {
                let state = RandomState::new();
                prop_assert_eq!(state.hash_one(&other), state.hash_one(&ranged_bitmap));
            }

            if ranged_bitmap.is_empty() {
                prop_assert_eq!(
                    ranged_bitmap.cmp(&other),
                    if other.is_empty() {
                        Ordering::Equal
                    } else {
                        Ordering::Less
                    }
                );
            } else {
                match ranged_bitmap.cmp(&other) {
                    Ordering::Less => {
                        prop_assert!(
                            other.last_set().unwrap_or(BitmapIndex::MAX) > *range.end()
                                || (other.includes(&ranged_bitmap)
                                    && other.first_set().unwrap_or(BitmapIndex::MIN) < *range.start())
                        )
                    }
                    Ordering::Equal => prop_assert_eq!(&ranged_bitmap, &other),
                    Ordering::Greater => prop_assert!(!other.includes(&ranged_bitmap)),
                }
            }

            let (other_finite, other_infinite) = split_infinite_bitmap(other.clone());

            let mut ranged_and_other = other_finite
                .iter_set()
                .filter(|idx| range.contains(idx))
                .collect::<Bitmap>();
            if let Some(infinite) = &other_infinite {
                if !ranged_bitmap.is_empty() {
                    ranged_and_other.set_range(infinite.start.max(*range.start())..=*range.end());
                }
            }
            test_and_sub(&ranged_bitmap, &other, &ranged_and_other)?;

            let mut ranged_or_other = other.clone();
            ranged_or_other.set_range(range);
            prop_assert_eq!(&(&ranged_bitmap | &other), &ranged_or_other);
            prop_assert_eq!(&(ranged_bitmap.clone() | &other), &ranged_or_other);
            let mut buf = ranged_bitmap.clone();
            buf |= &other;
            prop_assert_eq!(&buf, &ranged_or_other);

            let ranged_xor_other = ranged_or_other - ranged_and_other;
            prop_assert_eq!(&(&ranged_bitmap ^ &other), &ranged_xor_other);
            prop_assert_eq!(&(ranged_bitmap.clone() ^ &other), &ranged_xor_other);
            let mut buf = ranged_bitmap.clone();
            buf ^= &other;
            prop_assert_eq!(buf, ranged_xor_other);

            test_bitmap_ref_binops(&ranged_bitmap, &other)?;
        }

        #[test]
        fn from_iterator((indices, indices_set) in index_vec_and_set()) {
            let bitmap = indices.iter().copied().collect::<Bitmap>();
            prop_assert_eq!(bitmap.weight(), Some(indices_set.len()));
            for idx in indices_set {
                prop_assert!(bitmap.is_set(idx));
            }
        }

        #[allow(clippy::redundant_clone)]
        #[test]
        fn arbitrary(bitmap: Bitmap) {
            // Test properties pertaining to first iterator output
            prop_assert_eq!(bitmap.first_set(), bitmap.iter_set().next());
            prop_assert_eq!(bitmap.first_unset(), bitmap.iter_unset().next());
            prop_assert_eq!(bitmap.is_empty(), bitmap.first_set().is_none());
            prop_assert_eq!(bitmap.is_full(), bitmap.first_unset().is_none());

            // Test iterator-wide properties
            fn test_iter(
                bitmap: &Bitmap,
                iter_set: impl Iterator<Item = BitmapIndex>,
            ) -> Result<(Bitmap, String), TestCaseError> {
                let mut iter_set = iter_set.peekable();
                let mut iter_unset = bitmap.iter_unset().peekable();

                // Iterate over BitmapIndex until the end ot either iterator is reached
                let mut next_index = BitmapIndex::MIN;
                let mut set_stripe_start = None;
                let mut observed_weight = 0;
                let mut observed_last_set = None;
                let mut observed_last_unset = None;
                let mut inverse = Bitmap::full();
                let mut display = String::new();
                //
                while let (Some(next_set), Some(next_unset)) =
                    (iter_set.peek().copied(), iter_unset.peek().copied())
                {
                    // Move least advanced iterator forward
                    match next_set.cmp(&next_unset) {
                        Ordering::Less => {
                            // Next index should be set
                            iter_set.next();
                            prop_assert_eq!(next_set, next_index);

                            // Acknowledge that a set index has been processed
                            observed_last_set = Some(next_set);
                            observed_weight += 1;
                            if set_stripe_start.is_none() {
                                set_stripe_start = Some(next_set);
                            }
                        }
                        // Next index should be unset
                        Ordering::Greater => {
                            // Next index should be unset
                            iter_unset.next();
                            prop_assert_eq!(next_unset, next_index);
                            observed_last_unset = Some(next_unset);

                            // If we just went through a stripe of set indices,
                            // propagate that into the inverse & display predictions
                            if let Some(first_set) = set_stripe_start {
                                let last_set = observed_last_set.unwrap();
                                inverse.unset_range(first_set..=last_set);
                                if !display.is_empty() {
                                    write!(display, ",").unwrap();
                                }
                                write!(display, "{first_set}").unwrap();
                                if last_set != first_set {
                                    write!(display, "-{last_set}").unwrap();
                                }
                                set_stripe_start = None;
                            }
                        }
                        Ordering::Equal => unreachable!("Next index can't be both set and unset"),
                    }

                    // Update next_index
                    next_index = next_index.checked_add_signed(1).expect(
                        "Shouldn't overflow if we had both a next set & unset index before iterating",
                    );
                }

                // At this point, we reached the end of one of the iterators, and
                // the other iterator should just keep producing an infinite
                // sequence of consecutive indices. Reach some conclusions...
                let mut infinite_iter: Box<dyn Iterator<Item = BitmapIndex>> =
                    match (iter_set.peek(), iter_unset.peek()) {
                        (Some(next_set), None) => {
                            // Check end-of-iterator properties
                            prop_assert_eq!(bitmap.last_set(), None);
                            prop_assert_eq!(bitmap.last_unset(), observed_last_unset);
                            prop_assert_eq!(bitmap.weight(), None);

                            // Handle last (infinite) range of set elements
                            let stripe_start = set_stripe_start.unwrap_or(*next_set);
                            inverse.unset_range(stripe_start..);
                            if !display.is_empty() {
                                write!(display, ",").unwrap();
                            }
                            write!(display, "{stripe_start}-").unwrap();

                            // Expose infinite iterator of set elements
                            Box::new(iter_set)
                        }
                        (None, Some(_unset)) => {
                            // Check end-of-iterator properties
                            prop_assert_eq!(bitmap.last_set(), observed_last_set);
                            prop_assert_eq!(bitmap.last_unset(), None);
                            prop_assert_eq!(bitmap.weight(), Some(observed_weight));

                            // Handle previous range of set elements, if any
                            if let Some(first_set) = set_stripe_start {
                                let last_set = observed_last_set.unwrap();
                                inverse.unset_range(first_set..=last_set);
                                if !display.is_empty() {
                                    write!(display, ",").unwrap();
                                }
                                write!(display, "{first_set}").unwrap();
                                if last_set != first_set {
                                    write!(display, "-{last_set}").unwrap();
                                }
                            }

                            Box::new(iter_unset)
                        }
                        _ => unreachable!("At least one iterator is finite, they can't both be"),
                    };

                // ...and iterate the infinite iterator for a while to check it
                // does seem to meet expectations.
                for _ in 0..INFINITE_EXPLORE_ITERS {
                    prop_assert_eq!(infinite_iter.next(), Some(next_index));
                    if let Some(index) = next_index.checked_add_signed(1) {
                        next_index = index;
                    } else {
                        break;
                    }
                }

                // Return predicted bitmap inverse and display
                Ok((inverse, display))
            }
            let (inverse, display) = test_iter(&bitmap, (&bitmap).into_iter())?;
            let (inverse2, display2) = test_iter(&bitmap, bitmap.clone().into_iter())?;
            let (inverse3, display3) = test_iter(&bitmap, bitmap.iter_set())?;
            //
            prop_assert_eq!(&inverse, &inverse2);
            prop_assert_eq!(&inverse, &inverse3);
            prop_assert_eq!(&display, &display2);
            prop_assert_eq!(&display, &display3);
            prop_assert_eq!(&(!&bitmap), &inverse);
            prop_assert_eq!(&(!bitmap.clone()), &inverse);
            prop_assert_eq!(&format!("{bitmap:?}"), &display);
            prop_assert_eq!(&format!("{bitmap}"), &display);

            // Test properties that should be true of all bitmaps
            test_basic_inplace(&bitmap, &inverse)?;
            test_low_level_nonnull(bitmap.clone())?;

            // Test that a clone is indistinguishable from the original bitmap
            let clone = bitmap.clone();
            prop_assert_eq!(clone.first_set(), bitmap.first_set());
            prop_assert_eq!(clone.first_unset(), bitmap.first_unset());
            prop_assert_eq!(clone.is_empty(), bitmap.is_empty());
            prop_assert_eq!(clone.is_full(), bitmap.is_full());
            prop_assert_eq!(clone.last_set(), bitmap.last_set());
            prop_assert_eq!(clone.last_unset(), bitmap.last_unset());
            prop_assert_eq!(clone.weight(), bitmap.weight());
            //
            let (finite, infinite) = split_infinite_bitmap(bitmap);
            #[allow(clippy::option_if_let_else)]
            if let Some(infinite) = infinite {
                let test_iter = |mut iter_set: Box<dyn Iterator<Item = BitmapIndex>>| {
                    let mut iter_unset = clone.iter_unset().fuse();
                    let infinite_start = usize::from(infinite.start);
                    for idx in 0..infinite_start {
                        let next = if finite.is_set(idx) {
                            iter_set.next()
                        } else {
                            iter_unset.next()
                        };
                        prop_assert_eq!(next.map(usize::from), Some(idx));
                    }
                    prop_assert_eq!(iter_unset.next(), None);
                    for idx in (infinite_start..).take(INFINITE_EXPLORE_ITERS) {
                        prop_assert_eq!(iter_set.next().map(usize::from), Some(idx));
                    }
                    Ok(())
                };
                test_iter(Box::new((&clone).into_iter()))?;
                test_iter(Box::new(clone.iter_set()))?;
            } else {
                prop_assert_eq!(&((&clone).into_iter().collect::<Bitmap>()), &finite);
                prop_assert_eq!(&(clone.iter_set().collect::<Bitmap>()), &finite);

                let num_iters = usize::from(finite.last_set().unwrap_or(BitmapIndex::MIN)) + 1
                    - finite.weight().unwrap()
                    + INFINITE_EXPLORE_ITERS;
                let mut iterator = finite.iter_unset().zip(clone.iter_unset());
                for _ in 0..num_iters {
                    let (expected, actual) = iterator.next().unwrap();
                    prop_assert_eq!(expected, actual);
                }
            }
            //
            prop_assert_eq!(&format!("{clone:?}"), &display);
            prop_assert_eq!(&format!("{clone}"), &display);
            prop_assert_eq!(!&clone, inverse);
        }

        #[test]
        fn arbitrary_extend(
            bitmap: Bitmap,
            (extra, extra_set) in index_vec_and_set()
        ) {
            let mut extended = bitmap.clone();
            extended.extend(extra);

            if let Some(bitmap_weight) = bitmap.weight() {
                let extra_weight = extended
                    .weight()
                    .unwrap()
                    .checked_sub(bitmap_weight)
                    .expect("Extending a bitmap shouldn't reduce the weight");
                prop_assert!(extra_weight <= extra_set.len());
            }

            for idx in extra_set {
                prop_assert!(extended.is_set(idx));
            }
        }

        #[test]
        fn arbitrary_op_index(
            bitmap: Bitmap,
            index in bitmap_index()
        ) {
            test_indexing(&bitmap, index, bitmap.is_set(index))?;
        }

        #[test]
        fn arbitrary_op_range(
            bitmap: Bitmap,
            range in index_range()
        ) {
            let range_usize = range_to_usize(&range);
            let range_len = range_usize.clone().count();

            let mut buf = bitmap.clone();
            buf.set_range(range.clone());
            if let Some(bitmap_weight) = bitmap.weight() {
                let extra_weight = buf
                    .weight()
                    .unwrap()
                    .checked_sub(bitmap_weight)
                    .expect("Setting indices shouldn't reduce the weight");
                prop_assert!(extra_weight <= range_len);

                for idx in range_usize.clone() {
                    prop_assert!(buf.is_set(idx));
                }
            }

            buf.copy_from(&bitmap);
            buf.unset_range(range);
            if let Some(bitmap_weight) = bitmap.weight() {
                let lost_weight = bitmap_weight
                    .checked_sub(buf.weight().unwrap())
                    .expect("Clearing indices shouldn't increase the weight");
                prop_assert!(lost_weight <= range_len);

                for idx in range_usize {
                    prop_assert!(!buf.is_set(idx));
                }
            }
        }

        #[allow(clippy::similar_names)]
        #[test]
        fn arbitrary_op_bitmap([bitmap, other]: [Bitmap; 2]) {
            let (finite, infinite) = split_infinite_bitmap(bitmap.clone());
            let (other_finite, other_infinite) = split_infinite_bitmap(other.clone());

            prop_assert_eq!(
                bitmap.includes(&other),
                other_finite.iter_set().all(|idx| bitmap.is_set(idx))
                    && match (&infinite, &other_infinite) {
                        (Some(infinite), Some(other_infinite)) => {
                            (usize::from(other_infinite.start)..usize::from(infinite.start))
                                .all(|idx| finite.is_set(idx))
                        }
                        (_, None) => true,
                        (None, Some(_)) => false,
                    }
            );

            fn infinite_intersects_finite(infinite: &RangeFrom<BitmapIndex>, finite: &Bitmap) -> bool {
                finite
                    .last_set()
                    .map_or(false, |last_set| infinite.start <= last_set)
            }
            prop_assert_eq!(
                bitmap.intersects(&other),
                finite.iter_set().any(|idx| other.is_set(idx))
                    || match (&infinite, &other_infinite) {
                        (Some(_), Some(_)) => true,
                        (Some(infinite), None) => infinite_intersects_finite(infinite, &other_finite),
                        (None, Some(other_infinite)) =>
                            infinite_intersects_finite(other_infinite, &finite),
                        (None, None) => false,
                    }
            );

            prop_assert_eq!(
                bitmap == other,
                bitmap.includes(&other) && other.includes(&bitmap)
            );
            if bitmap == other {
                let state = RandomState::new();
                prop_assert_eq!(state.hash_one(&other), state.hash_one(&bitmap));
            }

            fn expected_cmp(bitmap: &Bitmap, reference: &Bitmap) -> Ordering {
                let (finite, infinite) = split_infinite_bitmap(bitmap.clone());
                let (ref_finite, ref_infinite) = split_infinite_bitmap(reference.clone());

                let finite_end = match (infinite, ref_infinite) {
                    (Some(_), None) => return Ordering::Greater,
                    (None, Some(_)) => return Ordering::Less,
                    (Some(infinite), Some(ref_infinite)) => infinite.start.max(ref_infinite.start),
                    (None, None) => finite
                        .last_set()
                        .unwrap_or(BitmapIndex::MIN)
                        .max(ref_finite.last_set().unwrap_or(BitmapIndex::MIN)),
                };

                for idx in (0..=usize::from(finite_end)).rev() {
                    match (bitmap.is_set(idx), reference.is_set(idx)) {
                        (true, false) => return Ordering::Greater,
                        (false, true) => return Ordering::Less,
                        _ => continue,
                    }
                }
                Ordering::Equal
            }
            prop_assert_eq!(bitmap.cmp(&other), expected_cmp(&bitmap, &other));

            let mut bitmap_and_other = finite
                .iter_set()
                .filter(|idx| other.is_set(*idx))
                .collect::<Bitmap>();
            match (&infinite, &other_infinite) {
                (Some(infinite), Some(other_infinite)) => {
                    bitmap_and_other.set_range(infinite.start.max(other_infinite.start)..);
                    for idx in usize::from(infinite.start)..usize::from(other_infinite.start) {
                        if other.is_set(idx) {
                            bitmap_and_other.set(idx);
                        }
                    }
                }
                (Some(infinite), None) => {
                    let other_end = other_finite.last_set().unwrap_or(BitmapIndex::MIN);
                    for idx in usize::from(infinite.start)..=usize::from(other_end) {
                        if other.is_set(idx) {
                            bitmap_and_other.set(idx)
                        }
                    }
                }
                _ => {}
            }
            test_and_sub(&bitmap, &other, &bitmap_and_other)?;

            let mut bitmap_or_other = finite;
            for idx in &other_finite {
                bitmap_or_other.set(idx);
            }
            if let Some(infinite) = infinite {
                bitmap_or_other.set_range(infinite);
            }
            if let Some(other_infinite) = other_infinite {
                bitmap_or_other.set_range(other_infinite);
            }
            prop_assert_eq!(&(&bitmap | &other), &bitmap_or_other);
            prop_assert_eq!(&(bitmap.clone() | &other), &bitmap_or_other);
            let mut buf = bitmap.clone();
            buf |= &other;
            prop_assert_eq!(&buf, &bitmap_or_other);

            let bitmap_xor_other = bitmap_or_other - bitmap_and_other;
            prop_assert_eq!(&(&bitmap ^ &other), &bitmap_xor_other);
            prop_assert_eq!(&(bitmap.clone() ^ &other), &bitmap_xor_other);
            buf.copy_from(&bitmap);
            buf ^= &other;
            prop_assert_eq!(buf, bitmap_xor_other);

            test_bitmap_ref_binops(&bitmap, &other)?;
        }
    }
}
