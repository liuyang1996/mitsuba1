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

#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/sampler.h>
#include <mitsuba/hw/basicshader.h>
#include "microfacet.h"
#include "ior.h"
#include <omp.h>
#include <mitsuba/core/plugin.h>
MTS_NAMESPACE_BEGIN

/*!\plugin{roughdielectric}{Rough dielectric material}
 * \order{5}
 * \icon{bsdf_roughdielectric}
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
 *     \parameter{intIOR}{\Float\Or\String}{Interior index of refraction specified
 *         numerically or using a known material name. \default{\texttt{bk7} / 1.5046}}
 *     \parameter{extIOR}{\Float\Or\String}{Exterior index of refraction specified
 *         numerically or using a known material name. \default{\texttt{air} / 1.000277}}
 *     \parameter{sampleVisible}{\Boolean}{
 *         Enables a sampling technique proposed by Heitz and D'Eon~\cite{Heitz1014Importance},
 *         which focuses computation on the visible parts of the microfacet normal
 *         distribution, considerably reducing variance in some cases.
 *         \default{\code{true}, i.e. use visible normal sampling}
 *     }
 *     \parameter{specular\showbreak Reflectance,\newline
 *         specular\showbreak Transmittance}{\Spectrum\Or\Texture}{Optional
 *         factor that can be used to modulate the specular reflection/transmission component. Note
 *         that for physical realism, this parameter should never be touched. \default{1.0}}
 * }\vspace{4mm}
 *
 * This plugin implements a realistic microfacet scattering model for rendering
 * rough interfaces between dielectric materials, such as a transition from air to
 * ground glass. Microfacet theory describes rough surfaces as an arrangement of
 * unresolved and ideally specular facets, whose normal directions are given by
 * a specially chosen \emph{microfacet distribution}. By accounting for shadowing
 * and masking effects between these facets, it is possible to reproduce the important
 * off-specular reflections peaks observed in real-world measurements of such
 * materials.
 * \renderings{
 *     \rendering{Anti-glare glass (Beckmann, $\alpha=0.02$)}
 *     	   {bsdf_roughdielectric_beckmann_0_0_2.jpg}
 *     \rendering{Rough glass (Beckmann, $\alpha=0.1$)}
 *     	   {bsdf_roughdielectric_beckmann_0_1.jpg}
 * }
 *
 * This plugin is essentially the ``roughened'' equivalent of the (smooth) plugin
 * \pluginref{dielectric}. For very low values of $\alpha$, the two will
 * be identical, though scenes using this plugin will take longer to render
 * due to the additional computational burden of tracking surface roughness.
 *
 * The implementation is based on the paper ``Microfacet Models
 * for Refraction through Rough Surfaces'' by Walter et al.
 * \cite{Walter07Microfacet}. It supports three different types of microfacet
 * distributions and has a texturable roughness parameter. Exterior and
 * interior IOR values can be specified independently, where ``exterior''
 * refers to the side that contains the surface normal. Similar to the
 * \pluginref{dielectric} plugin, IOR values can either be specified
 * numerically, or based on a list of known materials (see
 * \tblref{dielectric-iors} for an overview). When no parameters are given,
 * the plugin activates the default settings, which describe a borosilicate
 * glass BK7/air interface with a light amount of roughness modeled using a
 * Beckmann distribution.
 *
 * To get an intuition about the effect of the surface roughness parameter
 * $\alpha$, consider the following approximate classification: a value of
 * $\alpha=0.001-0.01$ corresponds to a material with slight imperfections
 * on an otherwise smooth surface finish, $\alpha=0.1$ is relatively rough,
 * and $\alpha=0.3-0.7$ is \emph{extremely} rough (e.g. an etched or ground
 * finish). Values significantly above that are probably not too realistic.
 *
 * Please note that when using this plugin, it is crucial that the scene contains
 * meaningful and mutually compatible index of refraction changes---see
 * \figref{glass-explanation} for an example of what this entails. Also, note that
 * the importance sampling implementation of this model is close, but
 * not always a perfect a perfect match to the underlying scattering distribution,
 * particularly for high roughness values and when the \texttt{ggx}
 * microfacet distribution is used. Hence, such renderings may
 * converge slowly.
 *
 * \subsubsection*{Technical details}
 * All microfacet distributions allow the specification of two distinct
 * roughness values along the tangent and bitangent directions. This can be
 * used to provide a material with a ``brushed'' appearance. The alignment
 * of the anisotropy will follow the UV parameterization of the underlying
 * mesh. This means that such an anisotropic material cannot be applied to
 * triangle meshes that are missing texture coordinates.
 *
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
 * When rendering with the Phong microfacet distribution, a conversion is
 * used to turn the specified Beckmann-equivalent $\alpha$ roughness value
 * into the exponents of the distribution. This is done in a way, such that
 * the different distributions all produce a similar appearance for the
 * same value of $\alpha$.
 *
 * \renderings{
 *     \rendering{Ground glass (GGX, $\alpha$=0.304,
 *     	   \lstref{roughdielectric-roughglass})}{bsdf_roughdielectric_ggx_0_304.jpg}
 *     \rendering{Textured roughness (\lstref{roughdielectric-textured})}
 *         {bsdf_roughdielectric_textured.jpg}
 * }
 *
 * \begin{xml}[caption=A material definition for ground glass, label=lst:roughdielectric-roughglass]
 * <bsdf type="roughdielectric">
 *     <string name="distribution" value="ggx"/>
 *     <float name="alpha" value="0.304"/>
 *     <string name="intIOR" value="bk7"/>
 *     <string name="extIOR" value="air"/>
 * </bsdf>
 * \end{xml}
 *
 * \begin{xml}[caption=A texture can be attached to the roughness parameter, label=lst:roughdielectric-textured]
 * <bsdf type="roughdielectric">
 *     <string name="distribution" value="beckmann"/>
 *     <float name="intIOR" value="1.5046"/>
 *     <float name="extIOR" value="1.0"/>
 *
 *     <texture name="alpha" type="bitmap">
 *         <string name="filename" value="roughness.exr"/>
 *     </texture>
 * </bsdf>
 * \end{xml}
 */
class RoughDielectric : public BSDF {
public:
	RoughDielectric(const Properties &props) : BSDF(props) {
		m_specularReflectance = new ConstantSpectrumTexture(
			props.getSpectrum("specularReflectance", Spectrum(1.0f)));
		m_specularTransmittance = new ConstantSpectrumTexture(
			props.getSpectrum("specularTransmittance", Spectrum(1.0f)));

		/* Specifies the internal index of refraction at the interface */
		Float intIOR = lookupIOR(props, "intIOR", "bk7");

		/* Specifies the external index of refraction at the interface */
		Float extIOR = lookupIOR(props, "extIOR", "air");

		if (intIOR < 0 || extIOR < 0 || intIOR == extIOR)
			Log(EError, "The interior and exterior indices of "
				"refraction must be positive and differ!");

		m_eta = intIOR / extIOR;
		m_invEta = 1 / m_eta;

		MicrofacetDistribution distr(props);
		m_type = distr.getType();
		m_sampleVisible = distr.getSampleVisible();

		m_alphaU = new ConstantFloatTexture(distr.getAlphaU());
		if (distr.getAlphaU() == distr.getAlphaV())
			m_alphaV = m_alphaU;
		else
			m_alphaV = new ConstantFloatTexture(distr.getAlphaV());
	}

	RoughDielectric(Stream *stream, InstanceManager *manager)
	 : BSDF(stream, manager) {
		m_type = (MicrofacetDistribution::EType) stream->readUInt();
		m_sampleVisible = stream->readBool();
		m_alphaU = static_cast<Texture *>(manager->getInstance(stream));
		m_alphaV = static_cast<Texture *>(manager->getInstance(stream));
		m_specularReflectance = static_cast<Texture *>(manager->getInstance(stream));
		m_specularTransmittance = static_cast<Texture *>(manager->getInstance(stream));
		m_eta = stream->readFloat();
		m_invEta = 1 / m_eta;

		configure();
	}

	void serialize(Stream *stream, InstanceManager *manager) const {
		BSDF::serialize(stream, manager);

		stream->writeUInt((uint32_t) m_type);
		stream->writeBool(m_sampleVisible);
		manager->serialize(stream, m_alphaU.get());
		manager->serialize(stream, m_alphaV.get());
		manager->serialize(stream, m_specularReflectance.get());
		manager->serialize(stream, m_specularTransmittance.get());
		stream->writeFloat(m_eta);
	}

	void configure() {
		unsigned int extraFlags = 0;
		if (m_alphaU != m_alphaV)
			extraFlags |= EAnisotropic;

		if (!m_alphaU->isConstant() || !m_alphaV->isConstant())
			extraFlags |= ESpatiallyVarying;

		m_components.clear();
		m_components.push_back(EGlossyReflection | EFrontSide
			| EBackSide | EUsesSampler | extraFlags
			| (m_specularReflectance->isConstant() ? 0 : ESpatiallyVarying));
		m_components.push_back(EGlossyTransmission | EFrontSide
			| EBackSide | EUsesSampler | ENonSymmetric | extraFlags
			| (m_specularTransmittance->isConstant() ? 0 : ESpatiallyVarying));

		/* Verify the input parameters and fix them if necessary */
		m_specularReflectance = ensureEnergyConservation(
			m_specularReflectance, "specularReflectance", 1.0f);
		m_specularTransmittance = ensureEnergyConservation(
			m_specularTransmittance, "specularTransmittance", 1.0f);

		m_usesRayDifferentials =
			m_alphaU->usesRayDifferentials() ||
			m_alphaV->usesRayDifferentials() ||
			m_specularReflectance->usesRayDifferentials() ||
			m_specularTransmittance->usesRayDifferentials();

		BSDF::configure();
	}


	float* brdfList(int res_wiU, int res_woU, int resWi, int resWo, int _seedIndex) const{

		int indexSeed = 12345678 + _seedIndex;;
		int sampleCount = resWi * resWi * resWo * resWo;

		int uniformSC = res_wiU * res_wiU * res_woU * res_woU;

		//we have both reflection and refraction
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
					for (int k = 0; k < 2; k++)
					{
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

									wo.z = (k == 0 ? sqrt(1.0 - wo.x * wo.x - wo.y*wo.y) : -sqrt(1.0 - wo.x * wo.x - wo.y*wo.y));
									Intersection its;
									BSDFSamplingRecord bRec(its, wi, wo, ERadiance);
									Spectrum value = eval(bRec, ESolidAngle);
									int index = uniformSC * 7 * k+ (indexWi * res_woU *res_woU + indexWo) * 7;
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
					} //which half sphere

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


	void deleteBRDFList(float* list)const
	{
		delete list;
		list = NULL;
	}

	Spectrum eval(const BSDFSamplingRecord &bRec, EMeasure measure) const {
		if (measure != ESolidAngle || Frame::cosTheta(bRec.wi) == 0)
			return Spectrum(0.0f);

		/* Determine the type of interaction */
		bool reflect = Frame::cosTheta(bRec.wi)
			* Frame::cosTheta(bRec.wo) > 0;

		Vector H;
		if (reflect) {
			/* Stop if this component was not requested */
			if ((bRec.component != -1 && bRec.component != 0)
				|| !(bRec.typeMask & EGlossyReflection))
				return Spectrum(0.0f);

			/* Calculate the reflection half-vector */
			H = normalize(bRec.wo+bRec.wi);
		} else {
			/* Stop if this component was not requested */
			if ((bRec.component != -1 && bRec.component != 1)
				|| !(bRec.typeMask & EGlossyTransmission))
				return Spectrum(0.0f);

			/* Calculate the transmission half-vector */
			Float eta = Frame::cosTheta(bRec.wi) > 0
				? m_eta : m_invEta;

			H = normalize(bRec.wi + bRec.wo*eta);
		}

		/* Ensure that the half-vector points into the
		   same hemisphere as the macrosurface normal */
		H *= math::signum(Frame::cosTheta(H));

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
		const Float F = fresnelDielectricExt(dot(bRec.wi, H), m_eta);

		/* Smith's shadow-masking function */
		const Float G = distr.G(bRec.wi, bRec.wo, H);

		if (reflect) {
			/* Calculate the total amount of reflection */
			Float value = F * D * G /
				(4.0f * std::abs(Frame::cosTheta(bRec.wi)));

			return m_specularReflectance->eval(bRec.its) * value;
		} else {
			Float eta = Frame::cosTheta(bRec.wi) > 0.0f ? m_eta : m_invEta;

			/* Calculate the total amount of transmission */
			Float sqrtDenom = dot(bRec.wi, H) + eta * dot(bRec.wo, H);
			Float value = ((1 - F) * D * G * eta * eta
				* dot(bRec.wi, H) * dot(bRec.wo, H)) /
				(Frame::cosTheta(bRec.wi) * sqrtDenom * sqrtDenom);

			/* Missing term in the original paper: account for the solid angle
			   compression when tracing radiance -- this is necessary for
			   bidirectional methods */
			Float factor = (bRec.mode == ERadiance)
				? (Frame::cosTheta(bRec.wi) > 0 ? m_invEta : m_eta) : 1.0f;

			return m_specularTransmittance->eval(bRec.its)
				* std::abs(value * factor * factor);
		}
	}

	Float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const {
		if (measure != ESolidAngle)
			return 0.0f;

		/* Determine the type of interaction */
		bool hasReflection   = ((bRec.component == -1 || bRec.component == 0)
							  && (bRec.typeMask & EGlossyReflection)),
		     hasTransmission = ((bRec.component == -1 || bRec.component == 1)
							  && (bRec.typeMask & EGlossyTransmission)),
		     reflect         = Frame::cosTheta(bRec.wi)
				             * Frame::cosTheta(bRec.wo) > 0;

		Vector H;
		Float dwh_dwo;

		if (reflect) {
			/* Zero probability if this component was not requested */
			if ((bRec.component != -1 && bRec.component != 0)
				|| !(bRec.typeMask & EGlossyReflection))
				return 0.0f;

			/* Calculate the reflection half-vector */
			H = normalize(bRec.wo+bRec.wi);

			/* Jacobian of the half-direction mapping */
			dwh_dwo = 1.0f / (4.0f * dot(bRec.wo, H));
		} else {
			/* Zero probability if this component was not requested */
			if ((bRec.component != -1 && bRec.component != 1)
				|| !(bRec.typeMask & EGlossyTransmission))
				return 0.0f;

			/* Calculate the transmission half-vector */
			Float eta = Frame::cosTheta(bRec.wi) > 0
				? m_eta : m_invEta;

			H = normalize(bRec.wi + bRec.wo*eta);

			/* Jacobian of the half-direction mapping */
			Float sqrtDenom = dot(bRec.wi, H) + eta * dot(bRec.wo, H);
			dwh_dwo = (eta*eta * dot(bRec.wo, H)) / (sqrtDenom*sqrtDenom);
		}

		/* Ensure that the half-vector points into the
		   same hemisphere as the macrosurface normal */
		H *= math::signum(Frame::cosTheta(H));

		/* Construct the microfacet distribution matching the
		   roughness values at the current surface position. */
		MicrofacetDistribution sampleDistr(
			m_type,
			m_alphaU->eval(bRec.its).average(),
			m_alphaV->eval(bRec.its).average(),
			m_sampleVisible
		);

		/* Trick by Walter et al.: slightly scale the roughness values to
		   reduce importance sampling weights. Not needed for the
		   Heitz and D'Eon sampling technique. */
		if (!m_sampleVisible)
			sampleDistr.scaleAlpha(1.2f - 0.2f * std::sqrt(
				std::abs(Frame::cosTheta(bRec.wi))));

		/* Evaluate the microfacet model sampling density function */
		Float prob = sampleDistr.pdf(math::signum(Frame::cosTheta(bRec.wi)) * bRec.wi, H);

		if (hasTransmission && hasReflection) {
			Float F = fresnelDielectricExt(dot(bRec.wi, H), m_eta);
			prob *= reflect ? F : (1-F);
		}

		return std::abs(prob * dwh_dwo);
	}

	Spectrum sample(BSDFSamplingRecord &bRec, const Point2 &_sample) const {
		Point2 sample(_sample);

		bool hasReflection = ((bRec.component == -1 || bRec.component == 0)
							  && (bRec.typeMask & EGlossyReflection)),
		     hasTransmission = ((bRec.component == -1 || bRec.component == 1)
							  && (bRec.typeMask & EGlossyTransmission)),
		     sampleReflection = hasReflection;

		if (!hasReflection && !hasTransmission)
			return Spectrum(0.0f);

		/* Construct the microfacet distribution matching the
		   roughness values at the current surface position. */
		MicrofacetDistribution distr(
			m_type,
			m_alphaU->eval(bRec.its).average(),
			m_alphaV->eval(bRec.its).average(),
			m_sampleVisible
		);

		/* Trick by Walter et al.: slightly scale the roughness values to
		   reduce importance sampling weights. Not needed for the
		   Heitz and D'Eon sampling technique. */
		MicrofacetDistribution sampleDistr(distr);
		if (!m_sampleVisible)
			sampleDistr.scaleAlpha(1.2f - 0.2f * std::sqrt(
				std::abs(Frame::cosTheta(bRec.wi))));

		/* Sample M, the microfacet normal */
		Float microfacetPDF;
		const Normal m = sampleDistr.sample(math::signum(Frame::cosTheta(bRec.wi)) * bRec.wi, sample, microfacetPDF);
		if (microfacetPDF == 0)
			return Spectrum(0.0f);

		Float cosThetaT;
		Float F = fresnelDielectricExt(dot(bRec.wi, m), cosThetaT, m_eta);
		Spectrum weight(1.0f);

		if (hasReflection && hasTransmission) {
			if (bRec.sampler->next1D() > F)
				sampleReflection = false;
		} else {
			weight = Spectrum(hasReflection ? F : (1-F));
		}

		if (sampleReflection) {
			/* Perfect specular reflection based on the microfacet normal */
			bRec.wo = reflect(bRec.wi, m);
			bRec.eta = 1.0f;
			bRec.sampledComponent = 0;
			bRec.sampledType = EGlossyReflection;

			/* Side check */
			if (Frame::cosTheta(bRec.wi) * Frame::cosTheta(bRec.wo) <= 0)
				return Spectrum(0.0f);

			weight *= m_specularReflectance->eval(bRec.its);
		} else {
			if (cosThetaT == 0)
				return Spectrum(0.0f);

			/* Perfect specular transmission based on the microfacet normal */
			bRec.wo = refract(bRec.wi, m, m_eta, cosThetaT);
			bRec.eta = cosThetaT < 0 ? m_eta : m_invEta;
			bRec.sampledComponent = 1;
			bRec.sampledType = EGlossyTransmission;

			/* Side check */
			if (Frame::cosTheta(bRec.wi) * Frame::cosTheta(bRec.wo) >= 0)
				return Spectrum(0.0f);

			/* Radiance must be scaled to account for the solid angle compression
			   that occurs when crossing the interface. */
			Float factor = (bRec.mode == ERadiance)
				? (cosThetaT < 0 ? m_invEta : m_eta) : 1.0f;

			weight *= m_specularTransmittance->eval(bRec.its) * (factor * factor);
		}

		if (m_sampleVisible)
			weight *= distr.smithG1(bRec.wo, m);
		else
			weight *= std::abs(distr.eval(m) * distr.G(bRec.wi, bRec.wo, m)
				* dot(bRec.wi, m) / (microfacetPDF * Frame::cosTheta(bRec.wi)));

		return weight;
	}

	Spectrum sample(BSDFSamplingRecord &bRec, Float &pdf, const Point2 &_sample) const {
		Point2 sample(_sample);

		bool hasReflection = ((bRec.component == -1 || bRec.component == 0)
							  && (bRec.typeMask & EGlossyReflection)),
		     hasTransmission = ((bRec.component == -1 || bRec.component == 1)
							  && (bRec.typeMask & EGlossyTransmission)),
		     sampleReflection = hasReflection;

		if (!hasReflection && !hasTransmission)
			return Spectrum(0.0f);

		/* Construct the microfacet distribution matching the
		   roughness values at the current surface position. */
		MicrofacetDistribution distr(
			m_type,
			m_alphaU->eval(bRec.its).average(),
			m_alphaV->eval(bRec.its).average(),
			m_sampleVisible
		);

		/* Trick by Walter et al.: slightly scale the roughness values to
		   reduce importance sampling weights. Not needed for the
		   Heitz and D'Eon sampling technique. */
		MicrofacetDistribution sampleDistr(distr);
		if (!m_sampleVisible)
			sampleDistr.scaleAlpha(1.2f - 0.2f * std::sqrt(
				std::abs(Frame::cosTheta(bRec.wi))));

		/* Sample M, the microfacet normal */
		Float microfacetPDF;
		const Normal m = sampleDistr.sample(math::signum(Frame::cosTheta(bRec.wi)) * bRec.wi, sample, microfacetPDF);
		if (microfacetPDF == 0)
			return Spectrum(0.0f);
		pdf = microfacetPDF;

		Float cosThetaT;
		Float F = fresnelDielectricExt(dot(bRec.wi, m), cosThetaT, m_eta);
		Spectrum weight(1.0f);

		if (hasReflection && hasTransmission) {
			if (bRec.sampler->next1D() > F) {
				sampleReflection = false;
				pdf *= 1-F;
			} else {
				pdf *= F;
			}
		} else {
			weight *= hasReflection ? F : (1-F);
		}

		Float dwh_dwo;
		if (sampleReflection) {
			/* Perfect specular reflection based on the microfacet normal */
			bRec.wo = reflect(bRec.wi, m);
			bRec.eta = 1.0f;
			bRec.sampledComponent = 0;
			bRec.sampledType = EGlossyReflection;

			/* Side check */
			if (Frame::cosTheta(bRec.wi) * Frame::cosTheta(bRec.wo) <= 0)
				return Spectrum(0.0f);

			weight *= m_specularReflectance->eval(bRec.its);

			/* Jacobian of the half-direction mapping */
			dwh_dwo = 1.0f / (4.0f * dot(bRec.wo, m));
		} else {
			if (cosThetaT == 0)
				return Spectrum(0.0f);

			/* Perfect specular transmission based on the microfacet normal */
			bRec.wo = refract(bRec.wi, m, m_eta, cosThetaT);
			bRec.eta = cosThetaT < 0 ? m_eta : m_invEta;
			bRec.sampledComponent = 1;
			bRec.sampledType = EGlossyTransmission;

			/* Side check */
			if (Frame::cosTheta(bRec.wi) * Frame::cosTheta(bRec.wo) >= 0)
				return Spectrum(0.0f);

			/* Radiance must be scaled to account for the solid angle compression
			   that occurs when crossing the interface. */
			Float factor = (bRec.mode == ERadiance)
				? (cosThetaT < 0 ? m_invEta : m_eta) : 1.0f;

			weight *= m_specularTransmittance->eval(bRec.its) * (factor * factor);

			/* Jacobian of the half-direction mapping */
			Float sqrtDenom = dot(bRec.wi, m) + bRec.eta * dot(bRec.wo, m);
			dwh_dwo = (bRec.eta*bRec.eta * dot(bRec.wo, m)) / (sqrtDenom*sqrtDenom);
		}

		if (m_sampleVisible)
			weight *= distr.smithG1(bRec.wo, m);
		else
			weight *= std::abs(distr.eval(m) * distr.G(bRec.wi, bRec.wo, m)
				* dot(bRec.wi, m) / (microfacetPDF * Frame::cosTheta(bRec.wi)));

		pdf *= std::abs(dwh_dwo);

		return weight;
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
			else if (name == "specularTransmittance")
				m_specularTransmittance = static_cast<Texture *>(child);
			else
				BSDF::addChild(name, child);
		} else {
			BSDF::addChild(name, child);
		}
	}

	Float getEta() const {
		return m_eta;
	}

	Float getRoughness(const Intersection &its, int component) const {
		return 0.5f * (m_alphaU->eval(its).average()
			+ m_alphaV->eval(its).average());
	}

	std::string toString() const {
		std::ostringstream oss;
		oss << "RoughDielectric[" << endl
			<< "  id = \"" << getID() << "\"," << endl
			<< "  distribution = " << MicrofacetDistribution::distributionName(m_type) << "," << endl
			<< "  sampleVisible = " << m_sampleVisible << "," << endl
			<< "  eta = " << m_eta << "," << endl
			<< "  alphaU = " << indent(m_alphaU->toString()) << "," << endl
			<< "  alphaV = " << indent(m_alphaV->toString()) << "," << endl
			<< "  specularReflectance = " << indent(m_specularReflectance->toString()) << "," << endl
			<< "  specularTransmittance = " << indent(m_specularTransmittance->toString()) << endl
			<< "]";
		return oss.str();
	}

	Shader *createShader(Renderer *renderer) const;

	MTS_DECLARE_CLASS()
private:
	MicrofacetDistribution::EType m_type;
	ref<Texture> m_specularTransmittance;
	ref<Texture> m_specularReflectance;
	ref<Texture> m_alphaU, m_alphaV;
	Float m_eta, m_invEta;
	bool m_sampleVisible;
};

/* Fake glass shader -- it is really hopeless to visualize
   this material in the VPL renderer, so let's try to do at least
   something that suggests the presence of a transparent boundary */
class RoughDielectricShader : public Shader {
public:
	RoughDielectricShader(Renderer *renderer, Float eta) :
		Shader(renderer, EBSDFShader) {
		m_flags = ETransparent;
	}

	Float getAlpha() const {
		return 0.3f;
	}

	void generateCode(std::ostringstream &oss,
			const std::string &evalName,
			const std::vector<std::string> &depNames) const {
		oss << "vec3 " << evalName << "(vec2 uv, vec3 wi, vec3 wo) {" << endl
			<< "    if (cosTheta(wi) < 0.0 || cosTheta(wo) < 0.0)" << endl
			<< "    	return vec3(0.0);" << endl
			<< "    return vec3(inv_pi * cosTheta(wo));" << endl
			<< "}" << endl
			<< endl
			<< "vec3 " << evalName << "_diffuse(vec2 uv, vec3 wi, vec3 wo) {" << endl
			<< "    return " << evalName << "(uv, wi, wo);" << endl
			<< "}" << endl;
	}


	MTS_DECLARE_CLASS()
};

Shader *RoughDielectric::createShader(Renderer *renderer) const {
	return new RoughDielectricShader(renderer, m_eta);
}

MTS_IMPLEMENT_CLASS(RoughDielectricShader, false, Shader)
MTS_IMPLEMENT_CLASS_S(RoughDielectric, false, BSDF)
MTS_EXPORT_PLUGIN(RoughDielectric, "Rough dielectric BSDF");
MTS_NAMESPACE_END
