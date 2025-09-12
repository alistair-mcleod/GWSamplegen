import numpy as np
import bilby.core.prior
from bilby.core.prior import (
    Cosine,
    PowerLaw,
    PriorDict,
    Sine,
    Uniform,
    Triangular,
)
from typing import Union
from bilby.gw.prior import UniformComovingVolume, UniformSourceFrame

def constructPrior(
	prior: Union[Uniform, Cosine, UniformComovingVolume, PowerLaw, UniformSourceFrame], 
	min: float, 
	max: float,
	**kwargs
) -> PriorDict:
	#generic constructor for bilby priors. 

	# if prior == PowerLaw:
	#     kwargs['alpha'] = powerlaw_alpha

	if max <= min:
		return max
	else:
		try:
			return prior(minimum = min, maximum = max, **kwargs)
		except:
			print("Failed to initialise prior with args, defaulting to no args.")
			return prior(minimum = min, maximum = max)
		
def inv_tri_bilby(mode,minimum,maximum,val):
	#copied from Bilby's triangular prior
	scale = maximum - minimum
	fractional_mode = (mode - minimum) / (maximum - minimum)

	below_mode = (val * scale * (mode - minimum)) ** 0.5
	above_mode = ((1 - val) * scale * (maximum - mode)) ** 0.5
	return (minimum + below_mode) * (val < fractional_mode) + (
		maximum - above_mode
	) * (val >= fractional_mode)

def tri_uniform(a,b, mode, r):
	#r = probability of drawing from uniform distribution
	#if r = 1, we draw from a uniform distribution
	#if r = 0, we draw from a triangular distribution
	p = np.random.uniform(0, 1)
	if p < r:
		return np.random.uniform(a,b)
	else:
		return inv_tri_bilby(mode, a, b, np.random.uniform(0, 1))

class TriUniform(bilby.core.prior.Prior):
	""" A combination of a uniform and triangular distribution
	
	Parameters
	----------
	minimum: float
		Minimum of the distribution
	maximum: float
		Maximum of the distribution
	mode: float	
		Mode of the triangular distribution
	r: float	
		Probability of drawing from the triangular distribution. With the default value of 2/3,
        the mode has twice the probability density of the uniform distribution.
	
	"""
	def __init__(self, minimum, maximum, mode, r = 2/3, name=None, latex_label=None, unit=None, boundary=None, **kwargs):
		self.mode = mode
		self.r = r
		if "alpha" in kwargs:
			#allow for 'alpha' as an alias for the r parameter
			self.r = kwargs['alpha']
		super(TriUniform, self).__init__(name=name, latex_label=latex_label, minimum=minimum, maximum=maximum, unit=unit, boundary=boundary)

	def rescale(self, val):
		return tri_uniform(self.minimum, self.maximum, self.mode, self.r)

	def sample(self, size=None):
		if size is None:
			return self.rescale(np.random.uniform(0, 1))
		else:
			return np.array([self.rescale(np.random.uniform(0, 1)) for i in range(size)])

	def __repr__(self):
		return f'TriUniform(minimum={self.minimum}, maximum={self.maximum}, mode={self.mode}, r={self.r})'


#construct a prior dictionary and set the priors for each parameter TODO: move to utils
def sample_masses_from_cm_q(parameters):
	converted = parameters.copy()
	converted['mass1_source'], converted['mass2_source'] = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_component_masses(parameters['chirp_mass'], parameters['mass_ratio'])
	#print(converted)
	return converted

# Function from LVC Rates & Populations Group
def powerlaw_setup(minv, maxv, alpha):
    a = (maxv / minv) ** (alpha + 1.) - 1.
    b = 1. / (alpha + 1.)
    return a, b


# Function from LVC Rates & Populations Group
def powerlaw_sample(x_rand, minv, a, b):
    return minv * (1. + a * x_rand) ** b


# Function from LVC Rates & Populations Group
def draw_mass_pair_power(m1_min, m1_max, m2_min, m2_max, m1pow, m2pow):
    a1, b1 = powerlaw_setup(m1_min, m1_max, m1pow)
    while True:
        x1 = np.random.random()
        x2 = np.random.random()
        m1 = powerlaw_sample(x1, m1_min, a1, b1)
        if m2_max < m1:
            a2, b2 = powerlaw_setup(m2_min, m2_max, m2pow)
        else:
            a2, b2 = powerlaw_setup(m2_min, m1, m2pow)
        m2 = powerlaw_sample(x2, m2_min, a2, b2)
        yield m1, m2



# Function from LVC Rates & Populations Group
def draw_spin_isotropic(spin_dict):
    '''
    Yields a random spin tuple (s_x, s_y, s_z) isotropically
    distributed with uniform magnitude distribution.
    '''
    max_spin = spin_dict

    while True:
        s = np.random.uniform(-1., 1., size=3)
        ssq = sum(s ** 2.)

        # s is a vector uniformly distributed inside the unit sphere
        # p(|s|) ~ |s|^2
        if ssq < 1.:
            # s * |s|^2 has magnitude |chi| = |s|^3
            # p(|chi|) d|chi| = p(|s|) d|s| ~ |s|^2 d|s| ~ const. d|chi|
            return s * ssq * max_spin