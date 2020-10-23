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

#include <mitsuba/core/fresolver.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/hw/basicshader.h>
#include "microfacet.h"
#include "ior.h"
#include <omp.h>
#include <mitsuba/core/plugin.h>

#include <openexr/ImfRgbaFile.h>
#include <openexr/ImfArray.h>
#include <openexr/ImfNamespace.h>
#include <openexr/OpenEXRConfig.h>

using namespace OPENEXR_IMF_NAMESPACE;

MTS_NAMESPACE_BEGIN

/*!\plugin{roughconductor}{Rough conductor material}
 * \order{7}
 * \icon{bsdf_roughconductor}
 * \parameters{
 *     \parameter{distribution}{\String}{
 *          Specifies the type of microfacet normal distribution
 *          used to model the surface roughness.
 *          \vspace{-1mm}
 *       \begin{enumerate}[(i)]
 *           \item \code{beckmann}: Physically-based distribution derived from
 *               Gaussian random surfaces. This is the default.\vspace{-1.5mm}
 *           \item \code{ggx}: The GGX \cite{Walter07Microfacet} distribution (also known as
 *               Trowbridge-Reitz \cite{Trowbridge19975Average} distribution)
 *               was designed to better approximate the long tails observed in measurements
 *               of ground surfaces, which are not modeled by the Beckmann distribution.
 *           \vspace{-1.5mm}
 *           \item \code{phong}: Anisotropic Phong distribution by
 *              Ashikhmin and Shirley \cite{Ashikhmin2005Anisotropic}.
 *              In most cases, the \code{ggx} and \code{beckmann} distributions
 *              should be preferred, since they provide better importance sampling
 *              and accurate shadowing/masking computations.
 *              \vspace{-4mm}
 *       \end{enumerate}
 *     }
 *     \parameter{alpha, alphaU, alphaV}{\Float\Or\Texture}{
 *         Specifies the roughness of the unresolved surface micro-geometry
 *         along the tangent and bitangent directions. When the Beckmann
 *         distribution is used, this parameter is equal to the
 *         \emph{root mean square} (RMS) slope of the microfacets.
 *         \code{alpha} is a convenience parameter to initialize both
 *         \code{alphaU} and \code{alphaV} to the same value. \default{0.1}.
 *     }
 *     \parameter{material}{\String}{Name of a material preset, see
 *           \tblref{conductor-iors}.\!\default{\texttt{Cu} / copper}}
 *     \parameter{eta, k}{\Spectrum}{Real and imaginary components of the material's index of
 *             refraction \default{based on the value of \texttt{material}}}
 *     \parameter{extEta}{\Float\Or\String}{
 *           Real-valued index of refraction of the surrounding dielectric,
 *           or a material name of a dielectric \default{\code{air}}
 *     }
 *     \parameter{sampleVisible}{\Boolean}{
 *         Enables a sampling technique proposed by Heitz and D'Eon~\cite{Heitz1014Importance},
 *         which focuses computation on the visible parts of the microfacet normal
 *         distribution, considerably reducing variance in some cases.
 *         \default{\code{true}, i.e. use visible normal sampling}
 *     }
 *     \parameter{specular\showbreak Reflectance}{\Spectrum\Or\Texture}{Optional
 *         factor that can be used to modulate the specular reflection component. Note
 *         that for physical realism, this parameter should never be touched. \default{1.0}}
 * }
 * \vspace{3mm}
 * This plugin implements a realistic microfacet scattering model for rendering
 * rough conducting materials, such as metals. It can be interpreted as a fancy
 * version of the Cook-Torrance model and should be preferred over
 * heuristic models like \pluginref{phong} and \pluginref{ward} if possible.
 * \renderings{
 *     \rendering{Rough copper (Beckmann, $\alpha=0.1$)}
 *     	   {bsdf_roughconductor_copper.jpg}
 *     \rendering{Vertically brushed aluminium (Anisotropic Phong,
 *         $\alpha_u=0.05,\ \alpha_v=0.3$), see
 *         \lstref{roughconductor-aluminium}}
 *         {bsdf_roughconductor_anisotropic_aluminium.jpg}
 * }
 *
 * Microfacet theory describes rough surfaces as an arrangement of unresolved
 * and ideally specular facets, whose normal directions are given by a
 * specially chosen \emph{microfacet distribution}. By accounting for shadowing
 * and masking effects between these facets, it is possible to reproduce the
 * important off-specular reflections peaks observed in real-world measurements
 * of such materials.
 *
 * This plugin is essentially the ``roughened'' equivalent of the (smooth) plugin
 * \pluginref{conductor}. For very low values of $\alpha$, the two will
 * be identical, though scenes using this plugin will take longer to render
 * due to the additional computational burden of tracking surface roughness.
 *
 * The implementation is based on the paper ``Microfacet Models
 * for Refraction through Rough Surfaces'' by Walter et al.
 * \cite{Walter07Microfacet}. It supports three different types of microfacet
 * distributions and has a texturable roughness parameter.
 * To facilitate the tedious task of specifying spectrally-varying index of
 * refraction information, this plugin can access a set of measured materials
 * for which visible-spectrum information was publicly available
 * (see \tblref{conductor-iors} for the full list).
 * There is also a special material profile named \code{none}, which disables
 * the computation of Fresnel reflectances and produces an idealized
 * 100% reflecting mirror.
 *
 * When no parameters are given, the plugin activates the default settings,
 * which describe copper with a medium amount of roughness modeled using a
 * Beckmann distribution.
 *
 * To get an intuition about the effect of the surface roughness parameter
 * $\alpha$, consider the following approximate classification: a value of
 * $\alpha=0.001-0.01$ corresponds to a material with slight imperfections
 * on an otherwise smooth surface finish, $\alpha=0.1$ is relatively rough,
 * and $\alpha=0.3-0.7$ is \emph{extremely} rough (e.g. an etched or ground
 * finish). Values significantly above that are probably not too realistic.
 * \vspace{4mm}
 * \begin{xml}[caption={A material definition for brushed aluminium}, label=lst:roughconductor-aluminium]
 * <bsdf type="roughconductor">
 *     <string name="material" value="Al"/>
 *     <string name="distribution" value="phong"/>
 *     <float name="alphaU" value="0.05"/>
 *     <float name="alphaV" value="0.3"/>
 * </bsdf>
 * \end{xml}
 *
 * \subsubsection*{Technical details}
 * All microfacet distributions allow the specification of two distinct
 * roughness values along the tangent and bitangent directions. This can be
 * used to provide a material with a ``brushed'' appearance. The alignment
 * of the anisotropy will follow the UV parameterization of the underlying
 * mesh. This means that such an anisotropic material cannot be applied to
 * triangle meshes that are missing texture coordinates.
 *
 * \label{sec:visiblenormal-sampling}
 * Since Mitsuba 0.5.1, this plugin uses a new importance sampling technique
 * contributed by Eric Heitz and Eugene D'Eon, which restricts the sampling
 * domain to the set of visible (unmasked) microfacet normals. The previous
 * approach of sampling all normals is still available and can be enabled
 * by setting \code{sampleVisible} to \code{false}.
 * Note that this new method is only available for the \code{beckmann} and
 * \code{ggx} microfacet distributions. When the \code{phong} distribution
 * is selected, the parameter has no effect.
 *
 * When rendering with the Phong microfacet distribution, a conversion is
 * used to turn the specified Beckmann-equivalent $\alpha$ roughness value
 * into the exponent parameter of this distribution. This is done in a way,
 * such that the same value $\alpha$ will produce a similar appearance across
 * different microfacet distributions.
 *
 * When using this plugin, you should ideally compile Mitsuba with support for
 * spectral rendering to get the most accurate results. While it also works
 * in RGB mode, the computations will be more approximate in nature.
 * Also note that this material is one-sided---that is, observed from the
 * back side, it will be completely black. If this is undesirable,
 * consider using the \pluginref{twosided} BRDF adapter.
 */
class RoughConductorApprox : public BSDF {
public:
	RoughConductorApprox(const Properties &props) : BSDF(props) {
		ref<FileResolver> fResolver = Thread::getThread()->getFileResolver();

		m_specularReflectance = new ConstantSpectrumTexture(
			props.getSpectrum("specularReflectance", Spectrum(1.0f)));

		m_R0 = props.getSpectrum("R0", Spectrum(1.0f));

		MicrofacetDistribution distr(props);
		m_type = distr.getType();
		m_sampleVisible = distr.getSampleVisible();

		m_alphaU = new ConstantFloatTexture(distr.getAlphaU());
		if (distr.getAlphaU() == distr.getAlphaV())
			m_alphaV = m_alphaU;
		else
			m_alphaV = new ConstantFloatTexture(distr.getAlphaV());
	}

	RoughConductorApprox(Stream *stream, InstanceManager *manager)
	 : BSDF(stream, manager) {
		m_type = (MicrofacetDistribution::EType) stream->readUInt();
		m_sampleVisible = stream->readBool();
		m_alphaU = static_cast<Texture *>(manager->getInstance(stream));
		m_alphaV = static_cast<Texture *>(manager->getInstance(stream));
		m_specularReflectance = static_cast<Texture *>(manager->getInstance(stream));
		m_R0 = Spectrum(stream);

		configure();
	}

	void serialize(Stream *stream, InstanceManager *manager) const {
		BSDF::serialize(stream, manager);

		stream->writeUInt((uint32_t) m_type);
		stream->writeBool(m_sampleVisible);
		manager->serialize(stream, m_alphaU.get());
		manager->serialize(stream, m_alphaV.get());
		manager->serialize(stream, m_specularReflectance.get());
		m_R0.serialize(stream);
	}

	void configure() {
		unsigned int extraFlags = 0;
		if (m_alphaU != m_alphaV)
			extraFlags |= EAnisotropic;

		if (!m_alphaU->isConstant() || !m_alphaV->isConstant() ||
			!m_specularReflectance->isConstant())
			extraFlags |= ESpatiallyVarying;

		m_components.clear();
		m_components.push_back(EGlossyReflection | EFrontSide | extraFlags);

		/* Verify the input parameters and fix them if necessary */
		m_specularReflectance = ensureEnergyConservation(
			m_specularReflectance, "specularReflectance", 1.0f);

		m_usesRayDifferentials =
			m_alphaU->usesRayDifferentials() ||
			m_alphaV->usesRayDifferentials() ||
			m_specularReflectance->usesRayDifferentials();

		BSDF::configure();
	}

	template <class T>
	inline T randUniform() const {
		return rand() / (T)RAND_MAX;
	}

	bool is_valid(const Vector2& projection) const {
		return projection.x * projection.x + projection.y * projection.y < 1.0;
	}


	float* brdfList(int res_wiU, int res_woU, int resWi, int resWo, int _seedIndex) const{

		int indexSeed = 12345678 + _seedIndex;;
		int sampleCount = resWi * resWi * resWo * resWo;

		int uniformSC = res_wiU * res_wiU * res_woU * res_woU;

		const int count = (sampleCount * 7 + uniformSC * 7)*2;

		float *result = new float[count];


		//memset(result, Color3(0.0f), resWi * resWi * resWo * resWo);

#if 1
		Properties propos("independent");
		propos.setInteger("seed", indexSeed);
		ref<Sampler> sampler = static_cast<Sampler *> (PluginManager::getInstance()->
			createObject(MTS_CLASS(Sampler), propos));


		sampler->generate(Point2i(0));

		int tcount = mts_omp_get_max_threads();
		std::vector<Sampler *> samplers(tcount);
		for (size_t i = 0; i<tcount; ++i) {
			ref<Sampler> clonedSampler = sampler->clone();
			clonedSampler->incRef();
			samplers[i] = clonedSampler.get();
		}
#endif
#pragma omp parallel for
		for (int i = 0; i < count; i++)
		{
			result[i] = 0.0f;
		}

		SLog(EDebug, "Start BRDF evaluation.");
		Intersection its;
		MicrofacetDistribution distr(
			m_type,
			m_alphaU->eval(its).average(),
			m_alphaV->eval(its).average(),
			m_sampleVisible
			);

#if 1
#pragma omp parallel for	
		for (int iwi = 0; iwi < res_wiU; iwi++)
		{
			for (int jwi = 0; jwi < res_wiU; jwi++)
			{
#if defined(MTS_OPENMP)
				int tid = mts_omp_get_thread_num();
#else
				int tid = 0;
#endif

				Sampler *sampler = samplers[tid];
				sampler->setSampleIndex(iwi);

				//generate a wi direction
				float random1 = sampler->next1D();// randUniform<Float>();
				float random2 = sampler->next1D();//  randUniform<Float>();
				Vector wi = Vector3((iwi + random1) / float(res_wiU) * 2 - 1, (jwi + random2) / float(res_wiU) * 2 - 1, 1.0);

				if (wi.x * wi.x + wi.y*wi.y < 1)
				{
					int indexWi = iwi * res_wiU + jwi;
					wi.z = sqrt(1.0 - wi.x * wi.x - wi.y*wi.y);
					//its.wi = wi;
					for (int iwo = 0; iwo < res_woU; iwo++)
					{
						for (int jwo = 0; jwo < res_woU; jwo++)
						{
							float random3 = sampler->next1D();// randUniform<Float>();
							float random4 = sampler->next1D();//  randUniform<Float>();
							Vector wo = Vector3((iwo + random3) / float(res_woU) * 2 - 1, (jwo + random4) / float(res_woU) * 2 - 1, 1.0);
							if (wo.x * wo.x + wo.y*wo.y < 1)
							{
								int indexWo = iwo * res_woU + jwo;

								wo.z = sqrt(1.0 - wo.x * wo.x - wo.y*wo.y);
								Intersection its;
								BSDFSamplingRecord bRec(its, wi, wo, ERadiance);
								Spectrum value = eval(bRec, ESolidAngle);
								int index = (indexWi * res_woU *res_woU + indexWo) * 7;
								result[index + 0] = (wi.x + 1) * 0.5;// Color3(value[0], value[1], value[2]);
								result[index + 1] = (wi.y + 1) * 0.5;
								result[index + 2] = (wo.x + 1) * 0.5;
								result[index + 3] = (wo.y + 1) * 0.5;// Color3(value[0], value[1], value[2]);

								result[index + 4] = value[0];
								result[index + 5] = value[1];
								result[index + 6] = value[2];
							}
						}
					}

				}
			}
		}
#endif

#if 0
		int startIndex = uniformSC * 7 * 2;

#pragma omp parallel for	
		for (int i = 0; i < sampleCount; i++)
		{

#if defined(MTS_OPENMP)
			int tid = mts_omp_get_thread_num();
#else
			int tid = 0;
#endif

			Sampler *sampler = samplers[tid];
			sampler->setSampleIndex(i);
			float pdf;

			//sample a half vector
			//float random1 = randUniform<Float>();
			//float random2 = randUniform<Float>();
			Normal m = distr.sampleAll(sampler->next2D(), pdf);

			//float random3 = randUniform<Float>();
			//float random4 = randUniform<Float>();

			float theta = sampler->next1D() * M_PI * 2;
			float z = sampler->next1D();// random4;

			Vector wi = Vector3(sqrt(1 - z*z) * cos(theta), sqrt(1 - z*z) * sin(theta), z);

			Vector wo = reflect(wi, m);

			Intersection its;
			BSDFSamplingRecord bRec(its, wi, wo, ERadiance);
			Spectrum value = eval(bRec, ESolidAngle);
			int index = i * 7;

			result[startIndex + index + 0] = (wi.x + 1) * 0.5;// Color3(value[0], value[1], value[2]);
			result[startIndex + index + 1] = (wi.y + 1) * 0.5;
			result[startIndex + index + 2] = (wo.x + 1) * 0.5;
			result[startIndex + index + 3] = (wo.y + 1) * 0.5;// Color3(value[0], value[1], value[2]);

			result[startIndex + index + 4] = value[0];
			result[startIndex + index + 5] = value[1];
			result[startIndex + index + 6] = value[2];

		}

#endif
#if 0
		//this is for test
		//output an image
		int res = 512;
		Array2D<Rgba> ndfData;
		ndfData.resizeErase(res, res);

		for (int i = 0; i < res; i++) //width
		{
			for (int j = 0; j < res; j++) //width
			{
				ndfData[j][i].r = 0;
				ndfData[j][i].g = 0;
				ndfData[j][i].b = 0;
				ndfData[j][i].a = 1.0f;
			}
		}
		Vector wi = Vector3(0, 0, 1);

		//#pragma omp parallel for	
		for (int i = 0; i < 1000000; i++)
		{

#if defined(MTS_OPENMP)
			int tid = mts_omp_get_thread_num();
#else
			int tid = 0;
#endif

			Sampler *sampler = samplers[tid];
			sampler->setSampleIndex(i);
			float pdf;

			//sample a half vector
			float random1 = randUniform<Float>();
			float random2 = randUniform<Float>();
			Normal m = distr.sampleAll(Point2(random1, random2), pdf);

			//Normal m = distr.sampleAll(sampler->next2D(), pdf);

			Vector wo = reflect(wi, m);

			Intersection its;
			BSDFSamplingRecord bRec(its, wi, wo, ERadiance);
			Spectrum value = eval(bRec, ESolidAngle);

			Vector h = wo;
			int jh = (h.y + 1) * res * 0.5;//(h.y + 1) * res * 0.5;
			int ih = (h.x + 1) * res * 0.5;//(h.x + 1) * res * 0.5;

			ndfData[jh][ih].r = value[0];
			ndfData[jh][ih].g = value[1];
			ndfData[jh][ih].b = value[2];
			ndfData[jh][ih].a = 1.0f;

		}

		for (int i = 0; i < res; i++) //width
		{
			for (int j = 0; j < res; j++) //width
			{
				//	int i = res / 2; int j = res / 2;
				const Vector2 h = Vector2(2.0f * i / res - 1.0, 2.0f * j / res - 1.0);// 1.0 - 2.0f * j / res);
				if (!is_valid(h))
				{
					ndfData[j][i].r = 0.5f;
					ndfData[j][i].g = 0.5f;
					ndfData[j][i].b = 0.5f;
					ndfData[j][i].a = 1.0f;
					continue;
				}
			}
		}

		cout << "Start ouput " << endl;
		std::string outputPath = "testEXR.exr";
		RgbaOutputFile file(outputPath.c_str(), res, res, WRITE_RGBA); // 1
		file.setFrameBuffer(&ndfData[0][0], 1, res); // 2
		file.writePixels(res); // 3
		cout << "Output Done " << endl;


#endif
		return result;
	}

	Spectrum fresnel_Schlick(const Vector3f &wi, const Vector3f &H) const {

		float temp = 1.0f - dot(wi, H);
		float temp_5 = pow(temp, 5);
		Spectrum result = m_R0 + (Spectrum(1.0f) - m_R0) * temp_5;
		return result;

	}


	void deleteBRDFList(float* list)const
	{
		delete list;
		list = NULL;
	}

	/// Helper function: reflect \c wi with respect to a given surface normal
	inline Vector reflect(const Vector &wi, const Normal &m) const {
		return 2 * dot(wi, m) * Vector(m) - wi;
	}

	Spectrum eval(const BSDFSamplingRecord &bRec, EMeasure measure) const {
		/* Stop if this component was not requested */
		if (measure != ESolidAngle ||
			Frame::cosTheta(bRec.wi) <= 0 ||
			Frame::cosTheta(bRec.wo) <= 0 ||
			((bRec.component != -1 && bRec.component != 0) ||
			!(bRec.typeMask & EGlossyReflection)))
			return Spectrum(0.0f);

		/* Calculate the reflection half-vector */
		Vector H = normalize(bRec.wo+bRec.wi);

		/* Construct the microfacet distribution matching the
		   roughness values at the current surface position. */
		MicrofacetDistribution distr(
			m_type,
			m_alphaU->eval(bRec.its).average(),
			m_alphaV->eval(bRec.its).average(),
			m_sampleVisible
		);

		/* Evaluate the microfacet normal distribution */
		const Float D = distr.eval(H);
		if (D == 0)
			return Spectrum(0.0f);

		/* Fresnel factor */
		const Spectrum F = fresnel_Schlick(bRec.wi, H) *  //fresnelConductorExact(dot(bRec.wi, H), m_eta, m_k) *
			m_specularReflectance->eval(bRec.its);

		/* Smith's shadow-masking function */
		const Float G = distr.G(bRec.wi, bRec.wo, H);

		/* Calculate the total amount of reflection */
		Float model = D * G / (4.0f * Frame::cosTheta(bRec.wi));

		return F * model;
	}

	Float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const {
		if (measure != ESolidAngle ||
			Frame::cosTheta(bRec.wi) <= 0 ||
			Frame::cosTheta(bRec.wo) <= 0 ||
			((bRec.component != -1 && bRec.component != 0) ||
			!(bRec.typeMask & EGlossyReflection)))
			return 0.0f;

		/* Calculate the reflection half-vector */
		Vector H = normalize(bRec.wo+bRec.wi);

		/* Construct the microfacet distribution matching the
		   roughness values at the current surface position. */
		MicrofacetDistribution distr(
			m_type,
			m_alphaU->eval(bRec.its).average(),
			m_alphaV->eval(bRec.its).average(),
			m_sampleVisible
		);

		if (m_sampleVisible)
			return distr.eval(H) * distr.smithG1(bRec.wi, H)
				/ (4.0f * Frame::cosTheta(bRec.wi));
		else
			return distr.pdf(bRec.wi, H) / (4 * absDot(bRec.wo, H));
	}

	Spectrum sample(BSDFSamplingRecord &bRec, const Point2 &sample) const {
		if (Frame::cosTheta(bRec.wi) < 0 ||
			((bRec.component != -1 && bRec.component != 0) ||
			!(bRec.typeMask & EGlossyReflection)))
			return Spectrum(0.0f);

		/* Construct the microfacet distribution matching the
		   roughness values at the current surface position. */
		MicrofacetDistribution distr(
			m_type,
			m_alphaU->eval(bRec.its).average(),
			m_alphaV->eval(bRec.its).average(),
			m_sampleVisible
		);

		/* Sample M, the microfacet normal */
		Float pdf;
		Normal m = distr.sample(bRec.wi, sample, pdf);

		if (pdf == 0)
			return Spectrum(0.0f);

		/* Perfect specular reflection based on the microfacet normal */
		bRec.wo = reflect(bRec.wi, m);
		bRec.eta = 1.0f;
		bRec.sampledComponent = 0;
		bRec.sampledType = EGlossyReflection;

		/* Side check */
		if (Frame::cosTheta(bRec.wo) <= 0)
			return Spectrum(0.0f);

		Spectrum F = fresnel_Schlick(bRec.wi, m) *//fresnelConductorExact(dot(bRec.wi, m),	m_eta, m_k) * 
			m_specularReflectance->eval(bRec.its);

		Float weight;
		if (m_sampleVisible) {
			weight = distr.smithG1(bRec.wo, m);
		} else {
			weight = distr.eval(m) * distr.G(bRec.wi, bRec.wo, m)
				* dot(bRec.wi, m) / (pdf * Frame::cosTheta(bRec.wi));
		}

		return F * weight;
	}

	Spectrum sample(BSDFSamplingRecord &bRec, Float &pdf, const Point2 &sample) const {
		if (Frame::cosTheta(bRec.wi) < 0 ||
			((bRec.component != -1 && bRec.component != 0) ||
			!(bRec.typeMask & EGlossyReflection)))
			return Spectrum(0.0f);

		/* Construct the microfacet distribution matching the
		   roughness values at the current surface position. */
		MicrofacetDistribution distr(
			m_type,
			m_alphaU->eval(bRec.its).average(),
			m_alphaV->eval(bRec.its).average(),
			m_sampleVisible
		);

		/* Sample M, the microfacet normal */
		Normal m = distr.sample(bRec.wi, sample, pdf);

		if (pdf == 0)
			return Spectrum(0.0f);

		/* Perfect specular reflection based on the microfacet normal */
		bRec.wo = reflect(bRec.wi, m);
		bRec.eta = 1.0f;
		bRec.sampledComponent = 0;
		bRec.sampledType = EGlossyReflection;

		/* Side check */
		if (Frame::cosTheta(bRec.wo) <= 0)
			return Spectrum(0.0f);

		Spectrum F = fresnel_Schlick(bRec.wi, m) *//fresnelConductorExact(dot(bRec.wi, m), m_eta, m_k) * 
			m_specularReflectance->eval(bRec.its);

		Float weight;
		if (m_sampleVisible) {
			weight = distr.smithG1(bRec.wo, m);
		} else {
			weight = distr.eval(m) * distr.G(bRec.wi, bRec.wo, m)
				* dot(bRec.wi, m) / (pdf * Frame::cosTheta(bRec.wi));
		}

		/* Jacobian of the half-direction mapping */
		pdf /= 4.0f * dot(bRec.wo, m);

		return F * weight;
	}

	void addChild(const std::string &name, ConfigurableObject *child) {
		if (child->getClass()->derivesFrom(MTS_CLASS(Texture))) {
			if (name == "alpha")
				m_alphaU = m_alphaV = static_cast<Texture *>(child);
			else if (name == "alphaU")
				m_alphaU = static_cast<Texture *>(child);
			else if (name == "alphaV")
				m_alphaV = static_cast<Texture *>(child);
			else if (name == "specularReflectance")
				m_specularReflectance = static_cast<Texture *>(child);
			else
				BSDF::addChild(name, child);
		} else {
			BSDF::addChild(name, child);
		}
	}

	Float getRoughness(const Intersection &its, int component) const {
		return 0.5f * (m_alphaU->eval(its).average()
			+ m_alphaV->eval(its).average());
	}

	std::string toString() const {
		std::ostringstream oss;
		oss << "RoughConductorApprox[" << endl
			<< "  id = \"" << getID() << "\"," << endl
			<< "  distribution = " << MicrofacetDistribution::distributionName(m_type) << "," << endl
			<< "  sampleVisible = " << m_sampleVisible << "," << endl
			<< "  alphaU = " << indent(m_alphaU->toString()) << "," << endl
			<< "  alphaV = " << indent(m_alphaV->toString()) << "," << endl
			<< "  specularReflectance = " << indent(m_specularReflectance->toString()) << "," << endl
			<< "  R0 = " << m_R0.toString() << endl
			<< "]";
		return oss.str();
	}

	Shader *createShader(Renderer *renderer) const;

	MTS_DECLARE_CLASS()
private:
	MicrofacetDistribution::EType m_type;
	ref<Texture> m_specularReflectance;
	ref<Texture> m_alphaU, m_alphaV;
	bool m_sampleVisible;
	Spectrum m_R0;
};

/**
 * GLSL port of the rough conductor shader. This version is much more
 * approximate -- it only supports the Ashikhmin-Shirley distribution,
 * does everything in RGB, and it uses the Schlick approximation to the
 * Fresnel reflectance of conductors. When the roughness is lower than
 * \alpha < 0.2, the shader clamps it to 0.2 so that it will still perform
 * reasonably well in a VPL-based preview.
 */
class RoughConductorApproxShader : public Shader {
public:
	RoughConductorApproxShader(Renderer *renderer, const Texture *specularReflectance,
			const Texture *alphaU, const Texture *alphaV, 
			const Spectrum &_R0) : Shader(renderer, EBSDFShader),
			m_specularReflectance(specularReflectance), m_alphaU(alphaU), m_alphaV(alphaV){
		m_specularReflectanceShader = renderer->registerShaderForResource(m_specularReflectance.get());
		m_alphaUShader = renderer->registerShaderForResource(m_alphaU.get());
		m_alphaVShader = renderer->registerShaderForResource(m_alphaV.get());

		/* Compute the reflectance at perpendicular incidence */
		m_R0 = _R0;////fresnelConductorExact(1.0f, eta, k);
	}

	bool isComplete() const {
		return m_specularReflectanceShader.get() != NULL &&
			   m_alphaUShader.get() != NULL &&
			   m_alphaVShader.get() != NULL;
	}

	void putDependencies(std::vector<Shader *> &deps) {
		deps.push_back(m_specularReflectanceShader.get());
		deps.push_back(m_alphaUShader.get());
		deps.push_back(m_alphaVShader.get());
	}

	void cleanup(Renderer *renderer) {
		renderer->unregisterShaderForResource(m_specularReflectance.get());
		renderer->unregisterShaderForResource(m_alphaU.get());
		renderer->unregisterShaderForResource(m_alphaV.get());
	}

	void resolve(const GPUProgram *program, const std::string &evalName, std::vector<int> &parameterIDs) const {
		parameterIDs.push_back(program->getParameterID(evalName + "_R0", false));
	}

	void bind(GPUProgram *program, const std::vector<int> &parameterIDs, int &textureUnitOffset) const {
		program->setParameter(parameterIDs[0], m_R0);
	}

	void generateCode(std::ostringstream &oss,
			const std::string &evalName,
			const std::vector<std::string> &depNames) const {
		oss << "uniform vec3 " << evalName << "_R0;" << endl
			<< endl
			<< "float " << evalName << "_D(vec3 m, float alphaU, float alphaV) {" << endl
			<< "    float ct = cosTheta(m), ds = 1-ct*ct;" << endl
			<< "    if (ds <= 0.0)" << endl
			<< "        return 0.0f;" << endl
			<< "    alphaU = 2 / (alphaU * alphaU) - 2;" << endl
			<< "    alphaV = 2 / (alphaV * alphaV) - 2;" << endl
			<< "    float exponent = (alphaU*m.x*m.x + alphaV*m.y*m.y)/ds;" << endl
			<< "    return sqrt((alphaU+2) * (alphaV+2)) * 0.15915 * pow(ct, exponent);" << endl
			<< "}" << endl
			<< endl
			<< "float " << evalName << "_G(vec3 m, vec3 wi, vec3 wo) {" << endl
			<< "    if ((dot(wi, m) * cosTheta(wi)) <= 0 || " << endl
			<< "        (dot(wo, m) * cosTheta(wo)) <= 0)" << endl
			<< "        return 0.0;" << endl
			<< "    float nDotM = cosTheta(m);" << endl
			<< "    return min(1.0, min(" << endl
			<< "        abs(2 * nDotM * cosTheta(wo) / dot(wo, m))," << endl
			<< "        abs(2 * nDotM * cosTheta(wi) / dot(wi, m))));" << endl
			<< "}" << endl
			<< endl
			<< "vec3 " << evalName << "_schlick(float ct) {" << endl
			<< "    float ctSqr = ct*ct, ct5 = ctSqr*ctSqr*ct;" << endl
			<< "    return " << evalName << "_R0 + (vec3(1.0) - " << evalName << "_R0) * ct5;" << endl
			<< "}" << endl
			<< endl
			<< "vec3 " << evalName << "(vec2 uv, vec3 wi, vec3 wo) {" << endl
			<< "   if (cosTheta(wi) <= 0 || cosTheta(wo) <= 0)" << endl
			<< "    	return vec3(0.0);" << endl
			<< "   vec3 H = normalize(wi + wo);" << endl
			<< "   vec3 reflectance = " << depNames[0] << "(uv);" << endl
			<< "   float alphaU = max(0.2, " << depNames[1] << "(uv).r);" << endl
			<< "   float alphaV = max(0.2, " << depNames[2] << "(uv).r);" << endl
			<< "   float D = " << evalName << "_D(H, alphaU, alphaV)" << ";" << endl
			<< "   float G = " << evalName << "_G(H, wi, wo);" << endl
			<< "   vec3 F = " << evalName << "_schlick(1-dot(wi, H));" << endl
			<< "   return reflectance * F * (D * G / (4*cosTheta(wi)));" << endl
			<< "}" << endl
			<< endl
			<< "vec3 " << evalName << "_diffuse(vec2 uv, vec3 wi, vec3 wo) {" << endl
			<< "    if (cosTheta(wi) < 0.0 || cosTheta(wo) < 0.0)" << endl
			<< "    	return vec3(0.0);" << endl
			<< "    return " << evalName << "_R0 * inv_pi * inv_pi * cosTheta(wo);"<< endl
			<< "}" << endl;
	}
	MTS_DECLARE_CLASS()
private:
	ref<const Texture> m_specularReflectance;
	ref<const Texture> m_alphaU;
	ref<const Texture> m_alphaV;
	ref<Shader> m_specularReflectanceShader;
	ref<Shader> m_alphaUShader;
	ref<Shader> m_alphaVShader;
	Spectrum m_R0;
};

Shader *RoughConductorApprox::createShader(Renderer *renderer) const {
	return new RoughConductorApproxShader(renderer,
		m_specularReflectance.get(), m_alphaU.get(), m_alphaV.get(), m_R0);
}

MTS_IMPLEMENT_CLASS(RoughConductorApproxShader, false, Shader)
MTS_IMPLEMENT_CLASS_S(RoughConductorApprox, false, BSDF)
MTS_EXPORT_PLUGIN(RoughConductorApprox, "Rough conductor BRDF");
MTS_NAMESPACE_END
