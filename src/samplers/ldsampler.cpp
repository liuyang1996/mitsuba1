/*
This file is part of Mitsuba, a physically based rendering system.

Copyright (c) 2007-2014 by Wenzel Jakob and others.

Mitsuba is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License Version 3
as published by the Free Software Foundation.

Mitsuba is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include <mitsuba/render/sampler.h>
#include <mitsuba/core/qmc.h>

MTS_NAMESPACE_BEGIN

/*!\plugin{ldsampler}{Low discrepancy sampler}
* \order{3}
* \parameters{
*     \parameter{sampleCount}{\Integer}{
*       Number of samples per pixel; should be a power of two
*       (e.g. 1, 2, 4, 8, 16, etc.), or it will be rounded up to the next one
*       \default{4}
*     }
*     \parameter{dimension}{\Integer}{
*       Effective dimension, up to which low discrepancy samples are provided. The
*       number here is to be interpreted as the number of subsequent 1D or 2D sample
*       requests that can be satisfied using ``good'' samples. Higher high values
*       increase both storage and computational costs.
*       \default{4}
*     }
* }
* \vspace{-2mm}
* \renderings{
*     \unframedrendering{A projection of the first 1024 points
*     onto the first two dimensions.}{sampler_ldsampler_0}
*     \unframedrendering{A projection of the first 1024 points
*     onto the 32 and 33th dimension, which look almost identical. However,
*     note that the points have been scrambled to reduce
*     correlations between dimensions.}{sampler_ldsampler_32}
* }
* This plugin implements a simple hybrid sampler that combines aspects of a Quasi-Monte
* Carlo sequence with a pseudorandom number generator based on a technique proposed
* by Kollig and Keller \cite{Kollig2002Efficient}.
* It is a good and fast general-purpose sample generator and therefore chosen as
* the default option in Mitsuba. Some of the QMC samplers in the following pages can generate
* even better distributed samples, but this comes at a higher cost in terms of performance.
*
* Roughly, the idea of this sampler is that all of the individual 2D sample dimensions are
* first filled using the same (0, 2)-sequence, which is then randomly scrambled and permuted
* using numbers generated by a Mersenne Twister pseudorandom number generator \cite{Saito2008SIMD}.
* Note that due to internal storage costs, low discrepancy samples are only provided
* up to a certain dimension, after which independent sampling takes over.
* The name of this plugin stems from the fact that (0, 2) sequences minimize the so-called
* \emph{star disrepancy}, which is a quality criterion on their spatial distribution. By
* now, the name has become slightly misleading since there are other samplers in Mitsuba
* that just as much try to minimize discrepancy, namely the \pluginref{sobol} and
* \pluginref{halton} plugins.
*
* Like the \pluginref{independent} sampler, multicore and network renderings
* will generally produce different images in subsequent runs due to the nondeterminism
* introduced by the operating system scheduler.
*/

class LowDiscrepancySampler : public Sampler {
public:
	LowDiscrepancySampler() : Sampler(Properties()) { }

	LowDiscrepancySampler(const Properties &props) : Sampler(props) {
		/* Sample count (will be rounded up to the next power of two) */
		m_sampleCount = props.getSize("sampleCount", 4);

		/* Dimension, up to which which low discrepancy samples are guaranteed to be available. */
		m_maxDimension = props.getInteger("dimension", 4);

		if (!math::isPowerOfTwo(m_sampleCount)) {
			m_sampleCount = math::roundToPowerOfTwo(m_sampleCount);
			Log(EWarn, "Sample count should be a power of two -- rounding to "
				SIZE_T_FMT, m_sampleCount);
		}

		m_samples1D = new Float*[m_maxDimension];
		m_samples2D = new Point2*[m_maxDimension];

		for (size_t i = 0; i<m_maxDimension; i++) {
			m_samples1D[i] = new Float[m_sampleCount];
			m_samples2D[i] = new Point2[m_sampleCount];
		}
		m_seedVal = props.getInteger("seed", -1);
		if (m_seedVal == -1)
			m_random = new Random();
		else
			m_random = new Random(m_seedVal);
	}

	LowDiscrepancySampler(Stream *stream, InstanceManager *manager)
		: Sampler(stream, manager) {
		m_random = static_cast<Random *>(manager->getInstance(stream));
		m_maxDimension = stream->readSize();

		m_samples1D = new Float*[m_maxDimension];
		m_samples2D = new Point2*[m_maxDimension];
		for (size_t i = 0; i<m_maxDimension; i++) {
			m_samples1D[i] = new Float[(size_t)m_sampleCount];
			m_samples2D[i] = new Point2[(size_t)m_sampleCount];
		}
	}

	virtual ~LowDiscrepancySampler() {
		for (size_t i = 0; i<m_maxDimension; i++) {
			delete[] m_samples1D[i];
			delete[] m_samples2D[i];
		}
		delete[] m_samples1D;
		delete[] m_samples2D;
	}

	void serialize(Stream *stream, InstanceManager *manager) const {
		Sampler::serialize(stream, manager);
		manager->serialize(stream, m_random.get());
		stream->writeSize(m_maxDimension);
	}

	ref<Sampler> clone() {
		ref<LowDiscrepancySampler> sampler = new LowDiscrepancySampler();

		sampler->m_sampleCount = m_sampleCount;
		sampler->m_maxDimension = m_maxDimension;
		sampler->m_random = new Random(m_random);
		sampler->m_samples1D = new Float*[m_maxDimension];
		sampler->m_samples2D = new Point2*[m_maxDimension];
		for (size_t i = 0; i<m_maxDimension; i++) {
			sampler->m_samples1D[i] = new Float[m_sampleCount];
			sampler->m_samples2D[i] = new Point2[m_sampleCount];
		}
		for (size_t i = 0; i<m_req1D.size(); ++i)
			sampler->request1DArray(m_req1D[i]);
		for (size_t i = 0; i<m_req2D.size(); ++i)
			sampler->request2DArray(m_req2D[i]);

		return sampler.get();
	}

	inline void generate1D(Float *samples, size_t sampleCount) {
#if defined(SINGLE_PRECISION)
		uint32_t scramble = m_random->nextULong() & 0xFFFFFFFF;
		for (size_t i = 0; i < sampleCount; ++i)
			samples[i] = radicalInverse2Single((uint32_t)i, scramble);
#else
		uint64_t scramble = m_random->nextULong();
		for (size_t i = 0; i < sampleCount; ++i)
			samples[i] = radicalInverse2Double(i, scramble);
#endif

		m_random->shuffle(samples, samples + sampleCount);
	}

	inline void generate2D(Point2 *samples, size_t sampleCount) {
#if defined(SINGLE_PRECISION)
		union {
			uint64_t qword;
			uint32_t dword[2];
		} scramble;

		scramble.qword = m_random->nextULong();

		for (size_t i = 0; i < sampleCount; ++i)
			samples[i] = sample02Single((uint32_t)i, scramble.dword);
#else
		uint64_t scramble[2];
		scramble[0] = m_random->nextULong();
		scramble[1] = m_random->nextULong();

		for (size_t i = 0; i < sampleCount; ++i)
			samples[i] = sample02Double(i, scramble);
#endif

		m_random->shuffle(samples, samples + sampleCount);
	}

	void generate(const Point2i &) {
		for (size_t i = 0; i<m_maxDimension; ++i) {
			generate1D(m_samples1D[i], m_sampleCount);
			generate2D(m_samples2D[i], m_sampleCount);
		}

		for (size_t i = 0; i<m_req1D.size(); i++)
			generate1D(m_sampleArrays1D[i], m_sampleCount * m_req1D[i]);

		for (size_t i = 0; i<m_req2D.size(); i++)
			generate2D(m_sampleArrays2D[i], m_sampleCount * m_req2D[i]);

		m_sampleIndex = 0;
		m_dimension1D = m_dimension2D = 0;
		m_dimension1DArray = m_dimension2DArray = 0;
	}

	void advance() {
		m_sampleIndex++;
		m_dimension1D = m_dimension2D = 0;
		m_dimension1DArray = m_dimension2DArray = 0;
	}

	void setSampleIndex(size_t sampleIndex) {
		m_sampleIndex = sampleIndex;
		m_dimension1D = m_dimension2D = 0;
		m_dimension1DArray = m_dimension2DArray = 0;
	}

	Float next1D() {
		Assert(m_sampleIndex < m_sampleCount);
		if (m_dimension1D < m_maxDimension)
			return m_samples1D[m_dimension1D++][m_sampleIndex];
		else
			return m_random->nextFloat();
	}

	Point2 next2D() {
		Assert(m_sampleIndex < m_sampleCount);
		if (m_dimension2D < m_maxDimension)
			return m_samples2D[m_dimension2D++][m_sampleIndex];
		else
			return Point2(m_random->nextFloat(), m_random->nextFloat());
	}

	std::string toString() const {
		std::ostringstream oss;
		oss << "LowDiscrepancySampler[" << endl
			<< "  sampleCount = " << m_sampleCount << "," << endl
			<< "  dimension = " << m_maxDimension << endl
			<< "]";
		return oss.str();
	}

	MTS_DECLARE_CLASS()
private:
	ref<Random> m_random;
	size_t m_maxDimension;
	size_t m_dimension1D;
	size_t m_dimension2D;
	Float **m_samples1D;
	Point2 **m_samples2D;
	uint64_t m_seedVal;
};

MTS_IMPLEMENT_CLASS_S(LowDiscrepancySampler, false, Sampler)
MTS_EXPORT_PLUGIN(LowDiscrepancySampler, "Low discrepancy sampler");
MTS_NAMESPACE_END
