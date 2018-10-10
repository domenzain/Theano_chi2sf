# Scalar Op whose c code depend on MIT code.

#The python code come from SciPy

import os

imported_scipy_special = False
try:
    import scipy.special
    import scipy.stats
    imported_scipy_special = True
# Importing scipy.special may raise ValueError.
# See http://projects.scipy.org/scipy/ticket/1739
except (ImportError, ValueError):
    pass

import theano
from theano import gof, tensor
from theano.scalar.basic import (BinaryScalarOp,
                                 upgrade_to_float,
                                 float_types)
from theano.scalar.basic_scipy import chi2sf
from theano.tensor.opt import register_specialize


class CChi2SF(BinaryScalarOp):
    """
    Compute (1 - chi2_cdf(x))
        ie. chi2 pvalue (chi2 'survival function')
    """

    @staticmethod
    def st_impl(x, k):
        return scipy.stats.chi2.sf(x, k)

    def impl(self, x, k):
        if imported_scipy_special:
            return Chi2SF.st_impl(x, k)
        else:
            super(Chi2SF, self).impl(x, k)

    def c_support_code(self):
        f = open(os.path.join(os.path.split(__file__)[0], 'gamma.c'))
        raw = f.read()
        return raw

    def c_code(self, node, name, inp, out, sub):
        x, k = inp
        z, = out
        if node.inputs[0].type in float_types:
            dtype = 'npy_' + node.outputs[0].dtype
            return """%(z)s =
                (%(dtype)s) 1 - GammaP(%(k)s/2., %(x)s/2.);""" % locals()
        raise NotImplementedError('only floatingpoint is implemented')

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))


cchi2sf = CChi2SF(upgrade_to_float, name='cchi2sf')


class GammaP(BinaryScalarOp):
    """
    Compute the regularized Gamma P function.
    """

    @staticmethod
    def st_impl(k, x):
        return scipy.stats.chi2.sf(k, x)

    def impl(self, k, x):
        if imported_scipy_special:
            return GammaP.st_impl(k, x)
        else:
            super(GammaP, self).impl(k, x)

    def c_support_code(self):
        f = open(os.path.join(os.path.split(__file__)[0], 'gamma.c'))
        raw = f.read()
        return raw

    def c_code(self, node, name, inp, out, sub):
        k, x = inp
        z, = out
        if node.inputs[0].type in float_types:
            dtype = 'npy_' + node.outputs[0].dtype
            return """%(z)s =
                (%(dtype)s) GammaP(%(k)s, %(x)s);""" % locals()
        raise NotImplementedError('only floatingpoint is implemented')

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))


gammap = GammaP(upgrade_to_float, name='gammap')


class GammaQ(BinaryScalarOp):
    """
    Compute the regularized Gamma Q function.
    """

    @staticmethod
    def st_impl(k, x):
        return scipy.stats.chi2.sf(x, k)

    def impl(self, k, x):
        if imported_scipy_special:
            return GammaQ.st_impl(k, x)
        else:
            super(GammaQ, self).impl(k, x)

    def c_support_code(self):
        f = open(os.path.join(os.path.split(__file__)[0], 'gamma.c'))
        raw = f.read()
        return raw

    def c_code(self, node, name, inp, out, sub):
        k, x = inp
        z, = out
        if node.inputs[0].type in float_types:
            dtype = 'npy_' + node.outputs[0].dtype
            return """%(z)s =
                (%(dtype)s) GammaQ(%(k)s, %(x)s);""" % locals()
        raise NotImplementedError('only floatingpoint is implemented')

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))


gammaq = GammaQ(upgrade_to_float, name='gammaq')


class GammaU(BinaryScalarOp):
    """
    Compute the upper incomplete Gamma function.
    """

    @staticmethod
    def st_impl(k, x):
        return scipy.stats.chi2.sf(x, k)

    def impl(self, k, x):
        if imported_scipy_special:
            return GammaU.st_impl(k, x)
        else:
            super(GammaU, self).impl(k, x)

    def c_support_code(self):
        f = open(os.path.join(os.path.split(__file__)[0], 'gamma.c'))
        raw = f.read()
        return raw

    def c_code(self, node, name, inp, out, sub):
        k, x = inp
        z, = out
        if node.inputs[0].type in float_types:
            dtype = 'npy_' + node.outputs[0].dtype
            return """%(z)s =
                (%(dtype)s) upperGamma(%(k)s, %(x)s);""" % locals()
        raise NotImplementedError('only floatingpoint is implemented')

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))


gammau = GammaU(upgrade_to_float, name='gammau')


class GammaL(BinaryScalarOp):
    """
    Compute the lower incomplete Gamma function.
    """

    @staticmethod
    def st_impl(k, x):
        return scipy.stats.chi2.sf(x, k)

    def impl(self, k, x):
        if imported_scipy_special:
            return GammaL.st_impl(k, x)
        else:
            super(GammaL, self).impl(k, x)

    def c_support_code(self):
        f = open(os.path.join(os.path.split(__file__)[0], 'gamma.c'))
        raw = f.read()
        return raw

    def c_code(self, node, name, inp, out, sub):
        k, x = inp
        z, = out
        if node.inputs[0].type in float_types:
            dtype = 'npy_' + node.outputs[0].dtype
            return """%(z)s =
                (%(dtype)s) lowerGamma(%(k)s, %(x)s);""" % locals()
        raise NotImplementedError('only floatingpoint is implemented')

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))


gammal = GammaL(upgrade_to_float, name='gammal')
