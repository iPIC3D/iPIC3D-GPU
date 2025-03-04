#ifndef _CUDA_GMM_KERNEL_H_
#define _CUDA_GMM_KERNEL_H_

#include <assert.h>

namespace cudaGMMWeight
{
    template <typename T, int dataDim, typename U>
    class GMMDataMultiDim;
}

namespace cudaGMMWeightKernel
{


/**
 * @brief calculate the log likelihood of the data points for all components, to be summed up to get the total log likelihood
 * @details this cuda kernel will be launched once for all data points
 * @param dataCUDAPtr pointer to the data, including numData
 * @param meanVector pointer to the mean vector, number of components * dataDim
 * @param coVarianceDecomp pointer to the decomposed coVariance matrix, lower triangular matrix, number of components * dataDim * dataDim
 * @param logLikelihoodForPoints pointer to the logLikelihoodForPoints array, to store the log p(x_i|mean,coVariance) of each data point, used later, number of components * numData
 * @param numComponents number of components
 */
template <typename T, int dataDim, typename U>
__global__ void calcLogLikelihoodForPointsKernel(const cudaGMMWeight::GMMDataMultiDim<T, dataDim, U>* dataCUDAPtr, const T* meanVector, const T* coVarianceDecomp, T* logLikelihoodForPoints, const int numComponents){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto numData = dataCUDAPtr->getNumData();
    if(idx >= numData)return;

    T xMinusMean[dataDim];  // (x - mean)
    T coVarianceNeg1TimesXMinusMean[dataDim]; // coVariance^-1 * (x - mean)

    for(int component = 0; component < numComponents; component++){
        auto meanComponent = meanVector + component*dataDim;
        auto coVarianceDecompComponent = coVarianceDecomp + component*dataDim*dataDim;
        auto logLikelihoods = logLikelihoodForPoints + component*numData; // p(x_i|mean,coVariance)
        
        T sum = 0;
        for(int dim = 0; dim < dataDim; dim++){
            // calculate (x - mean), dim
            xMinusMean[dim] = dataCUDAPtr->getDim(dim)[idx] - meanComponent[dim];           
        }

        for(int dim = 0; dim < dataDim; dim++){
            // slove lower triangular matrix
            sum = 0;
            // from head to tail
            if(dim > 0)
            {
                for(int j=0; j < dim-1; j++)
                {
                    sum += coVarianceDecompComponent[dim*dataDim + j] * coVarianceNeg1TimesXMinusMean[j];
                }   
            }

            coVarianceNeg1TimesXMinusMean[dim] = (xMinusMean[dim] - sum) / coVarianceDecompComponent[dim*dataDim + dim];
        }

        // slove the lower triangular matrix, transposed, it can be merged into the previous loop, but ...
        for(int dim=0; dim < dataDim; dim++){
            auto upperIndex = dataDim - dim - 1;
            sum = 0;
            // from tail to head
            for(int j=upperIndex+1; j < dataDim; j++)sum += coVarianceDecompComponent[j*dataDim + upperIndex] * coVarianceNeg1TimesXMinusMean[j];

            coVarianceNeg1TimesXMinusMean[upperIndex] = (coVarianceNeg1TimesXMinusMean[upperIndex] - sum) / coVarianceDecompComponent[upperIndex*dataDim + upperIndex];
        }

        T determinate = 1.0;
        sum = 0;
        for(int dim = 0; dim < dataDim; dim++){
            determinate *= coVarianceDecompComponent[dim*dataDim + dim];
            sum += coVarianceNeg1TimesXMinusMean[dim] * xMinusMean[dim];
        }
        determinate *= determinate;
        
        // calculate the log likelihood of this data point for this component
        logLikelihoods[idx] =  - 0.5 * (dataDim * log(2 * M_PI) + log(determinate)) - 0.5 * sum;
    }

}



/**
 * @brief calculate the log likelihood of the data points for all components, to be summed up to get the total log likelihood, and the posterior
 * @details this cuda kernel will be launched once for all data points
 * 
 * @param dataCUDAPtr pointer to the data, including numData
 * @param logWeightVector pointer to the weight vector, log(weight), number of components
 * @param logLikelihoodForPoints pointer to the log likelihood of the data points for all components, p(x_i|mean,coVariance), number of components * numData
 * @param logLikelihood pointer to the log likelihood(log p(x_i)) of the data points, to be summed up to get the total log likelihood(L or log p(x)), numData
 * @param posterior pointer to the posterior_nk(gamma) of the data points for all components, to be summed up to get the total Posterior(Gamma), number of components * numData
 */
template <typename T, int dataDim, typename U>
__global__ void calcLogLikelihoodPxAndposteriorKernel(const cudaGMMWeight::GMMDataMultiDim<T, dataDim, U>* dataCUDAPtr, const T* logWeightVector, const T* logLikelihoodForPoints, 
                                                        T* logLikelihood, T* posterior, const int numComponents){
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto numData = dataCUDAPtr->getNumData();
    if(idx >= numData)return;

    T maxValue = - INFINITY;
    T sum = 0;

    for(int component = 0; component < numComponents; component++){
        T logPxComponent = logWeightVector[component] + logLikelihoodForPoints[component*numData + idx]; // log(weight) + log(p(x_i|mean,coVariance))
        if(logPxComponent > maxValue)maxValue = logPxComponent;
    }

    for(int component = 0; component < numComponents; component++){
        T logPxComponent = logWeightVector[component] + logLikelihoodForPoints[component*numData + idx]; // log(weight) + log(p(x_i|mean,coVariance))
        sum += exp(logPxComponent - maxValue);
    }

    logLikelihood[idx] = maxValue + log(sum);   

    for(int component = 0; component < numComponents; component++){
        posterior[component*numData + idx] -= logLikelihood[idx];
    }

}


/**
 * @brief update the mean vector for each component
 * @details this cuda kernel will be launched once for all components, In one block. The shared memory should be sizeof(T) * numComponents
 * 
 * @param logWeightVector pointer to the old weight vector, log(weight), number of components
 * @param logPosterior pointer to the posterior_k(Gamma), number of components
 */
template <typename T>
__global__ void updateWeightKernel(T* logWeightVector, const T* logPosterior, const int numComponents){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numComponents)return;

    extern __shared__ T sharedLogMeanTimesPosterior[]; // for each component, the sum 
    sharedLogMeanTimesPosterior[threadIdx.x] = logWeightVector[idx] + logPosterior[idx]; // log(weight_k) + log(Posterior_k)
    __syncthreads();

    T sum = 0;
    for(int i = 0; i < numComponents; i++){
        sum += exp(sharedLogMeanTimesPosterior[i]);
    }

    logWeightVector[idx] = sharedLogMeanTimesPosterior[idx] - log(sum);
}


/**
 * @brief calculate the new coVariance matrix for each component, to be summed up and divided to get the total coVariance matrix
 * @details this cuda kernel will be launched once for all data points
 * 
 * @param dataCUDAPtr pointer to the data, including numData
 * @param logPosterior_nk pointer to the posterior_nk(Gamma), number of components * numData
 * @param logPosterior_k pointer to the posterior_k(Gamma), number of components
 * @param meanVector pointer to the mean vector, number of components * dataDim, just updated
 * @param tempCoVarianceForDataPoints pointer to the coVariance matrix for each data point, number of components * dataNum * dataDim * dataDim
 * 
 */
template <typename T, int dataDim, typename U>
__global__ void updateCoVarianceKernel(const cudaGMMWeight::GMMDataMultiDim<T, dataDim, U>* dataCUDAPtr, const T* logPosterior_nk, 
                                                                const T* logPosterior_k, const T* meanVector, 
                                                                T* tempCoVarianceForDataPoints, const int numComponents){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto numData = dataCUDAPtr->getNumData();
    if(idx >= numData)return;


    // for each component
    for(int component = 0; component < numComponents; component++){
        auto logPosterior_nkComponent = logPosterior_nk + component*numData;
        auto meanComponent = meanVector + component*dataDim;
        auto coVarianceComponent = tempCoVarianceForDataPoints + component*numData*dataDim*dataDim;

        // update the coVariance matrix
        T xMinusMean[dataDim];  // (x - mean^(t+1)) vector
        for(int dim = 0; dim < dataDim; dim++){
            xMinusMean[dim] = dataCUDAPtr->getDim(dim)[idx] - meanComponent[dim];
        }

        for(int i = 0; i < dataDim; i++){
            for(int j = 0; j < dataDim; j++){
                const auto elementInMatrix = i * dataDim + j;
                coVarianceComponent[elementInMatrix * numData + idx] = exp(logPosterior_nkComponent[idx]) * xMinusMean[i] * xMinusMean[j];
            }
        }

    }
}

/**
 * @brief decompose the coVariance matrix for each component
 * @details this cuda kernel will be launched once for all components
 * 
 * @param coVariance pointer to the coVariance matrix, number of components * dataDim * dataDim
 * @param coVarianceDecomp pointer to the decomposed coVariance matrix, lower triangular matrix, number of components * dataDim * dataDim
 * @param normalizer pointer to the normalizer, number of components
 */
template <typename T, int dataDim>
__global__ void decomposeCoVarianceKernel(const T* coVariance, T* coVarianceDecomp, T* normalizer, const int numComponents){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numComponents)return;

    auto coVarianceComponent = coVariance + idx*dataDim*dataDim;
    auto coVarianceDecompComponent = coVarianceDecomp + idx*dataDim*dataDim;  

    T logDeterminant = 0;

    for(int i = 0; i < dataDim*dataDim; i++){
        coVarianceDecompComponent[i] = 0;
    }

    // decompose the coVariance matrix
    for (int row = 0; row < dataDim; ++row) { // matrix row
        T sum = 0; // sum of left squared elements
        for (int j = 0; j < row; j++) {
            const T element = coVarianceDecompComponent[row * dataDim + j];
            sum += element * element;
        }
        assert(sum >= 0);
        sum = coVarianceComponent[row * dataDim + row] - sum;
        if (sum <= 0) { 
            assert(0);
            return;
        }

        coVarianceDecompComponent[row * dataDim + row] = sqrt(sum); // diagonal element
        logDeterminant += log(coVarianceDecompComponent[row * dataDim + row]);
        for (int i = row + 1; i < dataDim; ++i) 
        { // the row below the diagonal element
            T lowerElementSum = 0;
            for (int column = 0; column < row; column++)
                lowerElementSum += coVarianceDecompComponent[i * dataDim + column] * coVarianceDecompComponent[row * dataDim + column];

            coVarianceDecompComponent[i * dataDim + row] = (coVarianceComponent[i * dataDim + row] - lowerElementSum) / coVarianceDecompComponent[row * dataDim + row];
        }
    }

    logDeterminant *= 2;

    normalizer[idx] = - 0.5 * (dataDim * log(2.0 * M_PI) + logDeterminant);

}




/**
 * @brief check coVariance matrix elements for each component, ensure variance above a threshol eps and ensure coVariance matrix determinate > 0
 * @details this cuda kernel will be launched once for all components
 * 
 * @param coVariance pointer to the coVariance matrix, number of components * dataDim * dataDim
 * @param numComponents number of GMM components
 */
template <typename T, int dataDim>
__global__ void checkAdjustCoVarianceKernel(T* coVariance, const int numComponents){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numComponents)return;

    auto coVarianceComponent = coVariance + idx*dataDim*dataDim;

    constexpr T toll = 1e-12;
    constexpr T eps = 1e-7;

    // check NaN values or variance values too small
    // it works for any dataDim
    for(int i = 0; i<dataDim; i++ )
    {
        for(int j = 0; j<dataDim; j++)
        {
            if(std::isnan(coVarianceComponent[ i*dataDim + j]))
            {
                coVarianceComponent[ i*dataDim + j] = toll;
            }
        }
        if(coVarianceComponent[ i*dataDim + i] < eps )
        {
            coVarianceComponent[ i*dataDim + i] = eps;
        }
    }


    // from here on it works only if dataDim == 2

    // ensure symmetry in the cov-matrix 
    coVarianceComponent[1] = coVarianceComponent[2];

    // ensure determinate > 0
    if(coVarianceComponent[0]*coVarianceComponent[3] - coVarianceComponent[2]*coVarianceComponent[2] - toll <=0)
    {
        const T k = coVarianceComponent[3] / coVarianceComponent[0]; 
        coVarianceComponent[0] = sqrt( (coVarianceComponent[2]*coVarianceComponent[2] + toll) / k  ) + 10*eps;
        coVarianceComponent[3] = coVarianceComponent[0] * k;
    }
}




/**
 * @brief normalize data points such that the range is -1;+1 if normalizeBack == false - normalize data points such that the range is the original one if normalizeBack == true
 * @details this cuda kernel will be launched once for all data points
 * 
 * @param dataCUDAPtr pointer to the data, including numData
 * @param rescaleFactor pointer to the rescale factor, dataDim
 */
template <typename T, int dataDim, typename U, bool normalizeBack>
__global__ void normalizePointsKernel(cudaGMMWeight::GMMDataMultiDim<T, dataDim, U>* dataCUDAPtr, const T* rescaleFactor){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto numData = dataCUDAPtr->getNumData();
    if(idx >= numData)return;

    if constexpr(normalizeBack)
    {
        for(int dim = 0; dim < dataDim; dim++){
            // normalize data points
            dataCUDAPtr->getDim(dim)[idx] *= rescaleFactor[dim];
        }
    }
    else
    {
        for(int dim = 0; dim < dataDim; dim++){
            // normalize data points
            dataCUDAPtr->getDim(dim)[idx] /= rescaleFactor[dim];
        }
    }
    
}

/**
 * @brief normalize mean vector and cov-matrix back such that the range is the original one
 * @details this cuda kernel will be launched once for all components
 * 
 * @param meanVector pointer to the mean vector, number of components * dataDim
 * @param coVariance pointer to the coVariance matrix, number of components * dataDim * dataDim
 * @param rescaleFactor pointer to the rescale factor, dataDim (here it is assumed that rescaleFactor is homogenues in all dimensions --> to fix later) 
 */
template <typename T, int dataDim>
__global__ void normalizeMeanAndCovBack(T* meanVector, T* coVariance, const int numComponents, const T* rescaleFactor){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numComponents)return;

    auto meanComponent = meanVector + idx*dataDim;
    auto coVarianceComponent = coVariance + idx*dataDim*dataDim;

    for(int i=0; i< dataDim; i++)
    {
        meanComponent[i] *= rescaleFactor[i];
    }

    // to fix and make it general
    const T rescalFactorSqrd = rescaleFactor[0]*rescaleFactor[0]; 
    for(int i=0; i< dataDim*dataDim; i++)
    {
        coVarianceComponent[i] *= rescalFactorSqrd;
    }

}

}

#endif // _CUDA_GMM_KERNEL_H_