// DCP LoopCorrection device functions for runtime compilation.
// Source: https://bitbucket.org/pusuluri_krishna/deterministicchaosprospector/

#define N_EQ1	3
#define MAX_KNEADING_LENGTH 2001
#define NUM_THREADS_PER_BLOCK 512
#define INFINITY 100000
__device__ void stepper(const double* y, double* dydt, const double* params)
{
	double a=params[0], b = params[1], c=params[2];
	dydt[0] = -y[1] - y[2];
	dydt[1] = y[0] + a * y[1];
	dydt[2] = b * y[0] + y[2] * (y[0] - c );
}
__device__ void computeFirstDerivative(const double *y, double *dydt, const double* params){
        stepper(y, dydt, params);
        return;
}
__device__ double LZ76(bool * s, int n) {
  int c=1,l=1,i=0,k=1,kmax = 1,stop=0;
  while(stop ==0) {
    if (s[i+k-1] != s[l+k-1]) {
      if (k > kmax) {
        kmax=k;
      }
      i++;
      if (i==l) {
        c++;
        l += kmax;
        if (l+1>n)
          stop = 1;
        else {
          i=0;
          k=1;
          kmax=1;
        }
      } else {
        k=1;
      }
    } else {
      k++;
      if (l+k > n) {
        c++;
        stop =1;
      }
    }
  }
  return double(c)/n;
}
__device__ double  computePeriodNormalizedKneadingSum(bool *kneadings, unsigned kneadingsLength, unsigned periodLength ){
    double kneadingSum=0, minPeriodSum=0, currPeriodSum=0;
    unsigned i=0, normalizedPeriodIndex=0;
    
    double minPeriodSumSymmetric=0, currPeriodSumSymmetric=0;
    unsigned normalizedPeriodIndexSymmetric=0;
    if(periodLength<kneadingsLength){ 
        for(i=0; i<periodLength; i++) {
            currPeriodSum=0;
            for(unsigned j=0; j<periodLength; j++) {
                currPeriodSum+= 2*currPeriodSum + kneadings[i+j];
            }
            if(minPeriodSum==0 || currPeriodSum < minPeriodSum) {
                minPeriodSum = currPeriodSum;
                normalizedPeriodIndex=i;
            }
        }
        for(i=0; i<periodLength; i++) {
            currPeriodSumSymmetric=0;
            for(unsigned j=0; j<periodLength; j++) {
                currPeriodSumSymmetric+= 2*currPeriodSumSymmetric + 1-kneadings[i+j];
            }
            if(minPeriodSumSymmetric==0 || currPeriodSumSymmetric < minPeriodSumSymmetric) {
                minPeriodSumSymmetric = currPeriodSumSymmetric;
                normalizedPeriodIndexSymmetric=i;
            }
        }
    }
    for(i=0; i<kneadingsLength; i++) {
        if(minPeriodSum < minPeriodSumSymmetric){
            kneadingSum = kneadingSum + kneadings[normalizedPeriodIndex+periodLength-1-i%periodLength]/pow(2.,double(-i+kneadingsLength));

        } else {
            kneadingSum = kneadingSum + (1-kneadings[normalizedPeriodIndexSymmetric+periodLength-1-i%periodLength])/pow(2.,double(-i+kneadingsLength));

        }
    }
    if(periodLength<kneadingsLength){
    } 
    return kneadingSum;
}
__device__ double getNormalizedPeriodAndKneadingSum(bool* kneadings, unsigned kneadingsLength){
    bool periodFound=true;
    unsigned periodLength = kneadingsLength;
    for(unsigned currPeriod=1; currPeriod < kneadingsLength/2; currPeriod++) {
        periodFound=true;
        for(unsigned i=currPeriod; i < kneadingsLength-currPeriod; i+=currPeriod) {
            for ( unsigned j=0; j<currPeriod; j++) {
                if(kneadings[j] != kneadings[i+j]) {
                    periodFound=false;
                    break;
                }
            }
            if(!periodFound) {
                break;
            }
        }
        if(periodFound) {
            periodLength = currPeriod;
            break;
        }
    }
    return periodLength==kneadingsLength? LZ76(kneadings, kneadingsLength) : -computePeriodNormalizedKneadingSum( kneadings, kneadingsLength, periodLength);
}
__device__ double integrator_rk4(double* y_current, const double* params, const double dt, const unsigned N, const unsigned stride,
                                            const unsigned kneadingsStart, const unsigned kneadingsEnd)
{
	unsigned i, j, k, kneadingIndex=0, kneadingArrayIndex=0;
	double dt2, dt6;
	double y1[N_EQ1], y2[N_EQ1], k1[N_EQ1], k2[N_EQ1], k3[N_EQ1], k4[N_EQ1];
	double firstDerivativeCurrent[N_EQ1],firstDerivativePrevious=0;
	double kneadingsWeightedSum=0;
	bool kneadings[MAX_KNEADING_LENGTH];
	double a=params[0], b = params[1], c=params[2];
    bool currentDerivative1Direction=false, previousDerivate1Direction = false;
	dt2 = dt/2.; dt6 = dt/6.;
	for(i=1; i<N; i++)
	{
		for(j=0; j<stride; j++)
		{
			stepper(y_current, k1, params);
			for(k=0; k<N_EQ1; k++) y1[k] = y_current[k]+k1[k]*dt2;
			stepper(y1, k2, params);
			for(k=0; k<N_EQ1; k++) y2[k] = y_current[k]+k2[k]*dt2;
			stepper(y2, k3, params);
			for(k=0; k<N_EQ1; k++) y2[k] = y_current[k]+k3[k]*dt;
			stepper(y2, k4, params);
			for(k=0; k<N_EQ1; k++) y_current[k] += dt6*(k1[k]+2.*(k2[k]+k3[k])+k4[k]);
		}
		for(k=0; k<N_EQ1; k++) {
		    if(y_current[k]>INFINITY || y_current[k]<-INFINITY){
		        return -1.2;
		    }
        	}
		computeFirstDerivative(y_current, firstDerivativeCurrent, params);
		if(firstDerivativePrevious*firstDerivativeCurrent[0]<0 ){
            if(firstDerivativePrevious>0){
                currentDerivative1Direction = (firstDerivativeCurrent[1]>0);
                if(currentDerivative1Direction!=previousDerivate1Direction){
                    if(y_current[2] > 0.12*(c - a*b)/a){
                        if(kneadingIndex>=kneadingsStart){
                            kneadings[kneadingArrayIndex++]=1;
                        }
                    }else{
                        if(kneadingIndex>=kneadingsStart){
                            kneadings[kneadingArrayIndex++]=0;
                        }
                    }
                    kneadingIndex++;
                }
            }
            previousDerivate1Direction = (firstDerivativeCurrent[1]>0);
		}
		firstDerivativePrevious = firstDerivativeCurrent[0];
		if(kneadingIndex>kneadingsEnd)
			return getNormalizedPeriodAndKneadingSum(kneadings, kneadingArrayIndex);
	}
	return -1.1;
}

