[1mdiff --git a/src/samplers/independent.cpp b/src/samplers/independent.cpp[m
[1mindex cf43513..49f2974 100644[m
[1m--- a/src/samplers/independent.cpp[m
[1m+++ b/src/samplers/independent.cpp[m
[36m@@ -55,7 +55,11 @@[m [mpublic:[m
 	IndependentSampler(const Properties &props) : Sampler(props) {[m
 		/* Number of samples per pixel when used with a sampling-based integrator */[m
 		m_sampleCount = props.getSize("sampleCount", 4);[m
[31m-		m_random = new Random();[m
[32m+[m		[32mm_seedVal = props.getInteger("seed", -1);[m
[32m+[m		[32mif (m_seedVal == -1)[m
[32m+[m			[32mm_random = new Random();[m
[32m+[m		[32melse[m
[32m+[m			[32mm_random = new Random(m_seedVal);[m
 	}[m
 [m
 	IndependentSampler(Stream *stream, InstanceManager *manager)[m
[36m@@ -113,6 +117,7 @@[m [mpublic:[m
 	MTS_DECLARE_CLASS()[m
 private:[m
 	ref<Random> m_random;[m
[32m+[m	[32muint64_t m_seedVal;[m
 };[m
 [m
 MTS_IMPLEMENT_CLASS_S(IndependentSampler, false, Sampler)[m
