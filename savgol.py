import numpy as np

# This class produces square 2-dimensional Savitzky-Golay filtering kernels.
# 2D mono image data can be convolved with a kernel to compute smoothed version of the data or
# its partial derivatives of any order within the order of the polynomial.
#
# https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter
# https://scipy-cookbook.readthedocs.io/items/SavitzkyGolay.html
#
class SavitzkyGolay2D(object):
    def __init__(self, win_size, order):
        self.win_size = win_size = int(win_size)
        self.order = order = int(order)
        #
        if win_size % 2 == 0:
            raise ValueError('SavitzkyGolay2D: win_size must be odd (3, 5, 7, ...)')
        n_terms = (order + 1) * (order + 2)  / 2
        if win_size**2 < n_terms:
            raise ValueError('SavitzkyGolay2D: order is too high for win_size')
        half_size = win_size // 2
    
        # Exponents of the polynomial:
        # p(x,y) = a0 + a1*x + a2*y + a3*x^2 + a5*x*y + a4*y^2 + ...
        self.exps = exps = [(k-n, n) for k in range(order+1) for n in range(k+1)]
        # print('X-Y exponent pairs (', len(exps), '):', exps)
    
        # coordinates of points
        ind = np.arange(-half_size, half_size+1, dtype=np.float64)
        dy = np.repeat(ind, win_size)
        dx = np.tile(ind, [win_size, 1]).reshape(win_size**2,)
    
        # build matrix of system of equation
        A = np.empty( (win_size**2, len(exps)) )
        for i, exp in enumerate( exps ):
            A[:,i] = (dx**exp[0]) * (dy**exp[1])
        Astar = np.linalg.pinv(A)
        # print('Astar shape:', Astar.shape[0], 'x', Astar.shape[1])
        
        self.kernels = [Astar[i].reshape(win_size, -1) for i in range(Astar.shape[0])]
    # Return filtering kernel for dx's partial derivative along X and dy's PD along Y.
    # kernel(0, 0) - function itself (smoothed version)
    # kernel(1, 0) - first part derivative along X (d/dX)
    # kernel(0, 1) - first part derivative along Y (d/dY)
    # kernel(2, 0) - second PD along X (d^2/dX^2)
    # kernel(1, 1) - cross-PD (d^2/dXdY)
    #
    # The availability of higher-order derivatives is limited by the polynomial order. 
    def kernel(self, dx, dy):
        _exp = (dx, dy)
        for exp, kern in zip(self.exps, self.kernels):
            if exp == _exp:
                return kern
        return None
    # Combine d/dX and d/dY kernels into a single complex kernel.
    # Useful for computing 2D gradients with scipy.signal.convolve2d() as complex image data CD,
    # which can be translated to polar coordinates w. numpy.absolute(CD) and numpy.angle(CD).
    def gradKernel(self):
        dxk = self.kernel(1, 0)
        dyk = self.kernel(0, 1)
        if dxk is None or dyk is None:
            return None
        kern = np.empty(dxk.shape, dtype=complex)
        kern.real = dxk
        kern.imag = dyk
        return kern
#

if __name__ == '__main__':
    
    sg = SavitzkyGolay2D(5, 2)
    print (sg.kernel(0, 0))
    print ('')
    print (sg.kernel(0, 1))
    print ('')
    print (sg.kernel(1, 0))
