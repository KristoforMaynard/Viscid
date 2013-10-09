#!/usr/bin/env python
""" Fields belong in grids, or by themselves as a result of a calculation.
Important: The field must be able to reshape itself to the shape of its
coordinates, else there will be blood. Also, the order of the coords
matters, it is assumed that if the coords are z, y, x then the data
is iz, iy, ix. This can be permuted any which way, but order matters. """

from __future__ import print_function
import logging

import numpy as np

from . import coordinate
from . import vutil

LAYOUT_DEFAULT = "none"  # do not translate
LAYOUT_INTERLACED = "interlaced"
LAYOUT_FLAT = "flat"
LAYOUT_OTHER = "other"

def field_type_from_str(typ_str):
    for cls in vutil.subclass_spider(Field):
        if cls.TYPE == typ_str.lower():
            return cls
    logging.warn("Field type {0} not understood".format(typ_str))
    return None

def wrap_field(typ, name, crds, data, **kwargs):
    """ **kwargs passed to field constructor """
    #
    #len(clist), clist[0][0], len(clist[0][1]), type)
    cls = field_type_from_str(typ)
    if cls is not None:
        return cls(name, crds, data, **kwargs)
    else:
        raise NotImplementedError("can not decipher field")

def scalar_fields_to_vector(name, fldlist, **kwargs):
    if not name:
        name = fldlist[0].name
    center = fldlist[0].center
    crds = fldlist[0].crds
    time = fldlist[0].time
    # shape = fldlist[0].data.shape

    vfield = VectorField(name, crds, fldlist, center=center, time=time,
                         **kwargs)
    return vfield


class Field(object):
    TYPE = "none"
    CENTERING = ['node', 'cell', 'grid', 'face', 'edge']

    name = None  # String
    center = "none"  # String in CENTERING
    crds = None  # Coordinate object
    time = None  # float
    info = None  # dict

    source_data = None  # numpy-like object (h5py too)
    _cache = None  # this will always be a numpy array

    def __init__(self, name, crds, data, center="Node", time=0.0,
                 info=None, forget_source=False, **kwargs):
        self.name = name
        self.center = center.lower()
        self.time = time
        self.crds = crds
        self.data = data

        self.info = {} if info is None else info
        for k, v in kwargs.items():
            self.info[k] = v

        if forget_source:
            self.source_data = self.data

    def unload(self):
        """ does not guarentee that the memory will be freed """
        self._purge_cache()

    @property
    def dim(self):
        return self.crds.dim
        # try:
        #     return len(self.source_data.shape)
        # except AttributeError:
        #     # FIXME
        #     return len(self.source_data[0].shape) + 1

    @property
    def shape(self):
        # it is enforced that the cached data has a shape that agrees with
        # the coords by _reshape_ndarray_to_crds... actually, that method
        # requires this method to depend on the crd shape
        if self.center.lower() == "node":
            return list(self.crds.shape_nc)
        elif self.center.lower() == "cell":
            return list(self.crds.shape_cc)
        else:
            logging.warn("edge/face vectors not implemented, assuming "
                         "node shape")
            return self.crds.shape

    @property
    def size(self):
        return np.product(self.shape)

    @property
    def dtype(self):
        # print(type(self.source_data))
        if self._cache is not None:
            dt = self.source_data[0].dtype
            if isinstance(dt, str):
                return dt
            else:
                return self.source_data[0].dtype.name
        else:
            # dtype.name is for pruning endianness out of dtype
            if isinstance(self.source_data, (list, tuple)):
                return self.source_data[0].dtype.name
            else:
                return self.source_data.dtype.name

    @property
    def data(self):
        """ if you want to fill the cache, this will do it, note that
        to empty the cache later you can always use unload """
        if self._cache is None:
            self._fill_cache()
        return self._cache

    @data.setter
    def data(self, dat):
        self._purge_cache()
        self.source_data = dat

    def _purge_cache(self):
        """ does not guarentee that the memory will be freed """
        self._cache = None

    def _fill_cache(self):
        self._cache = self._translate_data(self.source_data)

    def _translate_data(self, dat):
        # some magic may need to happen here to accept more than np/h5 data
        # override if a type does something fancy (eg, interlacing)
        # and dat.flags["OWNDATA"]  # might i want this?
        return self._dat_to_ndarray(dat)

    def _dat_to_ndarray(self, dat):
        """ This should be the last thing called for all data that gets put
        into the cache. It makes dimensions jive correctly. This will translate
        non-ndarray data structures to a flat ndarray. Also, this function
        makes damn sure that the dimensionality of the coords matches the
        dimensionality of the array, which is to say, if the array is only 2d,
        but the coords have a 3rd dimension with length 1, reshape the array
        to include that extra dimension.
        """
        if isinstance(dat, np.ndarray):
            arr = dat
        else:
            # dtype.name is for pruning endianness out of dtype
            if isinstance(dat, (list, tuple)):
                dt = dat[0].dtype.name
                arr = np.array([np.array(d, dtype=dt) for d in dat], dtype=dt)

            elif isinstance(dat, Field):
                arr = dat.data
            else:
                arr = np.array(dat, dtype=dat.dtype.name)
        return self._reshape_ndarray_to_crds(arr)

    def _reshape_ndarray_to_crds(self, arr):
        """ enforce same dimensionality as coords here!
        self.shape better still be using the crds shape corrected for
        node / cell centering
        """
        if arr.shape == self.shape:
            return arr
        else:
            # print(">>> arr.shape: ", arr.shape)
            # print(">>> self.shape: ", self.shape)
            # print(len(self.crds["xcc"]), ", x = ", self.crds["xcc"])
            # print("center = ", self.center)
            return arr.reshape(self.shape)

    #TODO: some method that gracefully gets the correct crd arrays for
    # face and edge centered fields

    def _augment_slices(self, slices): #pylint: disable=R0201
        """ TODO: this is a crap mechanism to do vector slicing and should
        disappear """
        return slices

    def slice(self, selection, consolidate=False):
        """ Select a slice of the data using selection dictionary.
        Returns a new field.
        """
        cc = (self.center.lower() == "cell")
        slices, crdlst, reduced = self.crds.make_slice(selection, use_cc=cc,
                                                       consolidate=consolidate)

        # no slice necessary, just pass the field through
        if list(slices) == [slice(None)] * len(slices):
            return self

        crds = coordinate.wrap_crds(self.crds.TYPE, crdlst)
        slices = self._augment_slices(slices)
        # TODO: This can probably be done with a 'lazy slice'

        # if we sliced the hell out of the array, just
        # return the value that's left
        slced_dat = self.data[tuple(slices)]
        if len(reduced) == len(slices) or slced_dat.size == 1:
            return slced_dat
        else:
            fld = self.wrap(slced_dat,
                            {"name": self.name + "_slice",
                             "crds": crds,
                            })
            # if there are reduced dims, put them into the info dict
            if len(reduced) > 0:
                fld.info["reduced"] = reduced
            return fld

    def consolidate_dims(self):
        """ consolidate dimensions with length 1 in place """
        raise NotImplementedError()

    def n_points(self, center="none", **kwargs): #pylint: disable=W0613
        if center.lower() == "none":
            center = self.center
        return self.crds(center=center)

    def iter_points(self, center="none", **kwargs): #pylint: disable=W0613
        if center.lower() == "none":
            center = self.center
        return self.crds.iter_points(center=center)

    def __array__(self):
        return self.data

    def __enter__(self):
        return self

    def __exit__(self, typ, value, traceback):
        self.unload()
        return None

    def __iter__(self):
        for val in self.data.ravel():
            yield val

    def __getitem__(self, item):
        if isinstance(item, str):
            if item in self.crds:
                return self.crds[item]
            else:
                return self.slice(item)
        return self.slice(item)

    # def __getslice__(self, i, j):
    #     return self.data[slice(i, j)]

    ## emulate a numeric type
    def wrap(self, arr, context=None, typ=None):
        """ arr is the data to wrap... context is exta info to pass
        to the constructor. The return is just a number if arr is a
        1 element ndarray, this is for ufuncs that reduce to a scalar """
        if arr is NotImplemented:
            return NotImplemented
        # if just 1 number wrappen in an array, unpack the value and
        # return it... this is more ufuncy behavior
        if isinstance(arr, np.ndarray) and arr.size == 1:
            return np.ravel(arr)[0]
        if context is None:
            context = {}
        name = context.pop("name", self.name)
        crds = context.pop("crds", self.crds)
        center = context.pop("center", self.center)
        time = context.pop("time", self.time)
        # should it always return the same type as self?
        if typ is None:
            typ = type(self)
        elif isinstance(typ, str):
            typ = field_type_from_str(typ)
        return typ(name, crds, arr, time=time, center=center, info=context)

    def __array_wrap__(self, out_arr, context=None):
        # print("wrapping")
        return self.wrap(out_arr)

    # def __array_finalize__(self, *args, **kwargs):
    #     print("attempted call to field.__array_finalize__")

    def __add__(self, other):
        return self.wrap(self.data.__add__(other))
    def __sub__(self, other):
        return self.wrap(self.data.__sub__(other))
    def __mul__(self, other):
        return self.wrap(self.data.__mul__(other))
    def __div__(self, other):
        return self.wrap(self.data.__div__(other))
    def __truediv__(self, other):
        return self.wrap(self.data.__truediv__(other))
    def __floordiv__(self, other):
        return self.wrap(self.data.__floordiv__(other))
    def __mod__(self, other):
        return self.wrap(self.data.__mod__(other))
    def __divmod__(self, other):
        return self.wrap(self.data.__divmod__(other))
    def __pow__(self, other):
        return self.wrap(self.data.__pow__(other))
    def __lshift__(self, other):
        return self.wrap(self.data.__lshift__(other))
    def __rshift__(self, other):
        return self.wrap(self.data.__rshift__(other))
    def __and__(self, other):
        return self.wrap(self.data.__rshift__(other))
    def __xor__(self, other):
        return self.wrap(self.data.__rshift__(other))
    def __or__(self, other):
        return self.wrap(self.data.__rshift__(other))

    def __radd__(self, other):
        return self.wrap(self.data.__radd__(other))
    def __rsub__(self, other):
        return self.wrap(self.data.__rsub__(other))
    def __rmul__(self, other):
        return self.wrap(self.data.__rmul__(other))
    def __rdiv__(self, other):
        return self.wrap(self.data.__rdiv__(other))
    def __rtruediv__(self, other):
        return self.wrap(self.data.__rtruediv__(other))
    def __rfloordiv__(self, other):
        return self.wrap(self.data.__rfloordiv__(other))
    def __rmod__(self, other):
        return self.wrap(self.data.__rmod__(other))
    def __rdivmod__(self, other):
        return self.wrap(self.data.__rdivmod__(other))
    def __rpow__(self, other):
        return self.wrap(self.data.__rpow__(other))

    def __iadd__(self, other):
        raise NotImplementedError("Don't touch me like that")
    def __isub__(self, other):
        raise NotImplementedError("Don't touch me like that")
    def __imul__(self, other):
        raise NotImplementedError("Don't touch me like that")
    def __idiv__(self, other):
        raise NotImplementedError("Don't touch me like that")
    def __itruediv__(self, other):
        raise NotImplementedError("Don't touch me like that")
    def __ifloordiv__(self, other):
        raise NotImplementedError("Don't touch me like that")
    def __imod__(self, other):
        raise NotImplementedError("Don't touch me like that")
    def __ipow__(self, other):
        raise NotImplementedError("Don't touch me like that")

    def __neg__(self):
        return self.wrap(self.data.__neg__())
    def __pos__(self):
        return self.wrap(self.data.__pos__())
    def __abs__(self):
        return self.wrap(self.data.__abs__())
    def __invert__(self):
        return self.wrap(self.data.__invert__())

    def any(self):
        return self.data.any()
    def all(self):
        return self.data.all()

    def __lt__(self, other):
        return self.wrap(self.data.__lt__(other))
    def __le__(self, other):
        return self.wrap(self.data.__le__(other))
    def __eq__(self, other):
        return self.wrap(self.data.__eq__(other))
    def __ne__(self, other):
        return self.wrap(self.data.__ne__(other))
    def __gt__(self, other):
        return self.wrap(self.data.__gt__(other))
    def __ge__(self, other):
        return self.wrap(self.data.__ge__(other))


class ScalarField(Field):
    TYPE = "scalar"


class VectorField(Field):
    TYPE = "vector"

    _layout = None
    _ncomp = None
    _compdim = None

    def __init__(self, name, crds, data, **kwargs):
        forget_source = kwargs.pop("forget_source", False)
        super(VectorField, self).__init__(name, crds, data, **kwargs)
        if not "force_layout" in self.info:
            self.info["force_layout"] = LAYOUT_DEFAULT
        if forget_source:
            self.source_data = self.data


    def _purge_cache(self):
        """ does not guarentee that the memory will be freed """
        self._cache = None
        self._layout = None

    @property
    def ncomp(self):
        return self.data.shape[self.compdim]

    @property
    def compdim(self):
        """ dimension of the components of the vector """
        layout = self.layout
        if layout.lower() == LAYOUT_FLAT:
            return 0
        elif layout.lower() == LAYOUT_INTERLACED:
            return self.crds.dim
        elif layout.lower() == LAYOUT_OTHER:
            logging.warn("I don't know what your layout is, assuming vectors "
                         "are the last index (interleaved)...")
            return self.crds.dim

    @property
    def layout(self):
        # make sure that the data is translated before you inquire
        # about the layout
        if self._cache is None:
            self._fill_cache()
        return self._layout

    def _translate_data(self, dat):
        # if dat is list of fields, make it into a list of source_data so that
        # elements can be passed bare to np.array(...)
        if isinstance(dat, (list, tuple)):
            # dat = [d.source_data if isinstance(d, Field) else d for d in dat]
            for i in range(len(dat)):
                if isinstance(dat[i], Field):
                    # use source_data so that things don't get auto cached
                    # since we are caching own copy of the data anyway
                    dat[i] = dat[i].source_data  # pylint: disable=W0212

        dat_layout = self.detect_layout(dat)


        # we will preserve layout or we already have the correct layout,
        # do no translation... just like Field._translate_data
        if self.info["force_layout"].lower() == LAYOUT_DEFAULT or \
           self.info["force_layout"].lower() == dat_layout:
            self._layout = dat_layout
            return self._dat_to_ndarray(dat)

        # if layout is found to be other, i cant do anything with that
        elif dat_layout.lower() == LAYOUT_OTHER:
            logging.warn("Cannot auto-detect layout; not translating; "
                         "performance may suffer")
            self._layout = LAYOUT_OTHER
            return self._dat_to_ndarray(dat)

        # ok, we demand FLAT arrays, make it so
        elif self.info["force_layout"].lower() == LAYOUT_FLAT:
            if dat_layout != LAYOUT_INTERLACED:
                raise RuntimeError("should not be here")

            ncomp = dat.shape[-1]  # dat is interlaced
            dat_dest = np.empty([ncomp] + self.shape, dtype=dat.dtype.name)
            for i in range(ncomp):
                # NOTE: I wonder if this is the fastest way to reorder
                dat_dest[i, ...] = dat[..., i]
                # NOTE: no special case for lists, they are not
                # interpreted this way
            self._layout = LAYOUT_FLAT
            return self._dat_to_ndarray(dat_dest)

        # ok, we demand INTERLACED arrays, make it so
        elif self.info["force_layout"].lower() == LAYOUT_INTERLACED:
            if dat_layout != LAYOUT_FLAT:
                raise RuntimeError("should not be here")

            if isinstance(dat, (list, tuple)):
                ncomp = len(dat)
                dtype = dat[0].dtype.name
            else:
                ncomp = dat.shape[0]
                dtype = dat.dtype.name

            dat_dest = np.empty(self.shape + [ncomp], dtype=dtype)
            for i in range(ncomp):
                dat_dest[..., i] = dat[i]

            self._layout = LAYOUT_INTERLACED
            return self._dat_to_ndarray(dat_dest)

        # catch the remaining cases
        elif self.info["force_layout"].lower() == LAYOUT_OTHER:
            raise RuntimeError("How should I know how to force other layout?")
        else:
            raise ValueError("Bad argument for layout forcing")

    def _reshape_ndarray_to_crds(self, arr):
        """ enforce same dimensionality as coords here!
        self.shape better still be using the crds shape corrected for
        node / cell centering
        """
        target_shape = list(self.shape)
        # can't use self.ncomp or self.layout because we're in a weird
        # place and self.data hasn't been set yet, because this has to happen
        # first... like an ouroboros
        # NOTE: this logic is hideous, there must be a better way
        if self._layout.lower() == LAYOUT_FLAT:
            target_shape = [arr.shape[0]] + target_shape
        elif self._layout.lower() == LAYOUT_INTERLACED:
            target_shape = target_shape + [arr.shape[-1]]
        else:
            # assuming flat?
            target_shape = [arr.shape[0]] + target_shape

        if arr.shape == target_shape:
            return arr
        else:
            return arr.reshape(target_shape)

    def component_views(self):
        """ return numpy views to components individually, memory layout
        of the original field is maintained """
        ncomp = self.ncomp
        if self.layout.lower() == LAYOUT_FLAT:
            return [self.data[i, ...] for i in range(ncomp)]
        elif self.layout.lower() == LAYOUT_INTERLACED:
            return [self.data[..., i] for i in range(ncomp)]
        else:
            return [self.data[..., i] for i in range(ncomp)]

    def component_fields(self):
        n = self.name
        crds = self.crds
        c = self.center
        t = self.time
        views = self.component_views()
        lst = [None] * len(views)
        for i, v in enumerate(views):
            lst[i] = ScalarField("{0} {1}".format(n, i), crds, v, center=c,
                                 time=t)
        return lst

    def detect_layout(self, dat):
        """ returns LAYOUT_XXX """
        # if i receive a list, then i suppose i have a list of
        # arrays, one for each component... this is a flat layout
        if isinstance(dat, (list, tuple)):
            return LAYOUT_FLAT

        shape = self.shape

        # if the crds shape has more values than the dat.shape
        # then try trimmeng the directions that have 1 element
        # this can happen when crds are 3d, but field is only 2d
        while len(shape) > len(dat.shape) - 1:
            try:
                shape.remove(1)
            except ValueError:
                break

        if list(dat.shape[1:]) == shape:
            return LAYOUT_FLAT
        elif list(dat.shape[:-1]) == shape:
            return LAYOUT_INTERLACED
        elif dat.shape[0] == np.prod(shape):
            return LAYOUT_INTERLACED
        elif dat.shape[-1] == np.prod(shape):
            return LAYOUT_FLAT
        else:
            return LAYOUT_OTHER

    def _augment_slices(self, slices):
        """ TODO: this is a crap mechanism to do vector slicing and should
        disappear """
        slices.insert(self.compdim, slice(None))
        return slices


class MatrixField(Field):
    TYPE = "matrix"


class TensorField(Field):
    TYPE = "tensor"


##
## EOF
##
