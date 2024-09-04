//! Copy-on-write type analogous to [`std::borrow::Cow`] that can either be an
//! owned [`Bitmap`]/[`CpuSet`]/[`NodeSet`] or a [`BitmapRef`] thereof.

use super::{BitmapRef, OwnedBitmap};
#[cfg(doc)]
use crate::{bitmap::Bitmap, cpu::cpuset::CpuSet, memory::nodeset::NodeSet};
#[cfg(doc)]
use std::borrow::Cow;
use std::{
    borrow::Borrow,
    fmt::{self, Display, Formatter, Pointer},
    hash::{self, Hash},
    ops::Deref,
};

/// Clone-on-write smart pointer to a [`Bitmap`]-like `Target`
///
/// This is to [`Cow`] what [`BitmapRef`] is to `&Target`. It cannot literally
/// be a standard [`Cow`] because of the same C hwloc API peculiarities that
/// prevent [`BitmapRef`] from being a standard Rust reference.
///
/// You are not expected to use this type on a regular basis, it only exists to
/// serve the narrow use case of letting you return both owned and borrowed
/// bitmaps from topology-editing callbacks, without having to clone the
/// borrowed bitmaps. This is why its API is less extensive and convenient than
/// that of [`BitmapRef`].
#[derive(Clone, Debug)]
pub enum BitmapCow<'target, Target: OwnedBitmap> {
    /// Borrowed bitmap
    Borrowed(BitmapRef<'target, Target>),

    /// Owned bitmap
    Owned(Target),
}

impl<'target, Target: OwnedBitmap> BitmapCow<'target, Target> {
    /// Extracts the owned bitmap
    ///
    /// Clones the bitmap if it is not already owned.
    pub fn into_owned(self) -> Target {
        match self {
            Self::Borrowed(b) => b.clone_target(),
            Self::Owned(o) => o,
        }
    }
}

impl<'target, Target: OwnedBitmap> AsRef<Target> for BitmapCow<'target, Target> {
    fn as_ref(&self) -> &Target {
        match self {
            Self::Borrowed(b) => b.as_ref(),
            Self::Owned(o) => o,
        }
    }
}

impl<Target: OwnedBitmap> Borrow<Target> for BitmapCow<'_, Target> {
    fn borrow(&self) -> &Target {
        self.as_ref()
    }
}

impl<Target: OwnedBitmap> Deref for BitmapCow<'_, Target> {
    type Target = Target;

    fn deref(&self) -> &Target {
        self.as_ref()
    }
}

impl<Target: OwnedBitmap + Display> Display for BitmapCow<'_, Target> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        <Target as Display>::fmt(self.as_ref(), f)
    }
}

impl<Target: OwnedBitmap + Eq + PartialEq<Self>> Eq for BitmapCow<'_, Target> {}

impl<Target: OwnedBitmap> From<Target> for BitmapCow<'_, Target> {
    fn from(value: Target) -> Self {
        Self::Owned(value)
    }
}

impl<'target, Target: OwnedBitmap> From<&'target Target> for BitmapCow<'target, Target> {
    fn from(value: &'target Target) -> Self {
        Self::Borrowed(value.into())
    }
}

impl<'target, Target: OwnedBitmap> From<BitmapRef<'target, Target>> for BitmapCow<'target, Target> {
    fn from(value: BitmapRef<'target, Target>) -> Self {
        Self::Borrowed(value)
    }
}

impl<'target, Target: OwnedBitmap + Hash> Hash for BitmapCow<'target, Target> {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.as_ref().hash(state)
    }
}

impl<Target, Rhs> PartialEq<Rhs> for BitmapCow<'_, Target>
where
    Target: OwnedBitmap + PartialEq<Rhs>,
{
    fn eq(&self, other: &Rhs) -> bool {
        self.as_ref() == other
    }
}

impl<Target: OwnedBitmap> Pointer for BitmapCow<'_, Target> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        <Target as fmt::Pointer>::fmt(self.as_ref(), f)
    }
}
