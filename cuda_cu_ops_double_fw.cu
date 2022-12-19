// There is no double atomic add, so use atomicCAS
 __device__ inline void AtomicAddDouble( double *address, double value )
 {
    unsigned long long oldval, newval, readback; 
 
    oldval = __double_as_longlong(*address);
    newval = __double_as_longlong(__longlong_as_double(oldval) + value);
    while( (readback = atomicCAS((unsigned long long *)address, oldval, newval)) != oldval )
    {
        oldval = readback;
        newval = __double_as_longlong(__longlong_as_double(oldval) + value);
    }
 }

// LSMR algorithm (https://stanford.edu/group/SOL/software/lsmr)
// There is two computationaly expensive steps per algorithm iteration
// u = A * v * 1/alpha - (alpha/beta) * u;
// beta = norm(u)
// v = A^T * u * 1/beta - (beta/alpha) * v;
// alpha = norm(v)
//
// where A - sparse matrix, A^T - transposed A (precalculated), v and u are vectors, alpha and beta are norms of v and u

// Following CUDA function implements operation
// v3 = A * v1 * c1 - c2 * v2
// v3norm = norm(v3)

// Function parameters:
// data - matrix non-zero double values
// col_indices - column indices
// M - matrix rows number
// aM - offset to the next element of the row. Aligned to 128 bytes
// aMxN - total size of the arrays data and col_indices
// c1, c2 - double constants
// v2 - double vector
// v1 is loaded to texture v1Tex

 texture<int2, cudaTextureType1D, cudaReadModeElementType> v1Tex;
 __global__ void KernelMulMVC_VC_Doublefw( const double *data, const int *col_indices, const int M, const int aM, const int aMxN, const double c1, const double *v2, const double c2, double *v3, double *v3norm )
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int blockSize = blockDim.x;
    int row = bid * blockSize + tid;

    // Declare shared memory array
    extern __shared__ volatile double sdata[];

    // Return this thread is out of input matrix
    if( row >= M )
        return;

    // Iterate through one row of row-compressed matrix
    double sum = 0.0;
    for( int s = row; s < aMxN; s += aM )
    {
        int vidx = col_indices[s]; // Get column index
        if( vidx == -1 ) // Negative index means no more data in this row
            break;
        int2 vi = tex1Dfetch( v1Tex, vidx ); // Read double as int2 from texture
        double v = __hiloint2double( vi.y, vi.x ); // Convert int2 to double
        sum += data[s] * v;
    }

    // Calculate and output 1 component of the result vector
    double nv2 = sum * c1 - v2[row] * c2;
    v3[row] = nv2;

    // Calculate square sum of result vector components for this group
    sdata[tid] = nv2 * nv2; // Every thread calculates square of 1 component
    __syncthreads(); // Make sure all threads are done

    // Divide shared memory array in half and add values in second half to the values in the first half
    // repeat process until size 32
    for( int s = blockSize>>1; s > 32; s >>= 1 )
    {
        if( tid < s )
            sdata[tid] += sdata[tid + s];
        __syncthreads(); // Make sure all threads are done
    }

    // Don't need sychronization for single warp
    if( tid < 32 )
    {
        sdata[tid] += sdata[tid + 32];
        sdata[tid] += sdata[tid + 16];
        sdata[tid] += sdata[tid + 8];
        sdata[tid] += sdata[tid + 4];
        sdata[tid] += sdata[tid + 2];
        sdata[tid] += sdata[tid + 1];
    }

    // Zero thread atomically add this group result to the global squared norm
    if( tid == 0 )
        AtomicAddDouble( v3norm, sdata[0] );
}

void CUDACSRMV_Doublefw( int block_size, int rnum, int arnum, int cnum, int nznum, double *data, int *cind, double *v1, double c1, double *v2, double c2, double *v3, double *v3norm )
{
    cudaBindTexture( (size_t)0, v1Tex, v1, sizeof(double)*cnum );
    cudaDeviceSynchronize();
    
    int block_num = rnum / block_size + 1;
    KernelMulMVC_VC_Doublefw<<<block_num, block_size, block_size*sizeof(double)>>>( data, cind, rnum, arnum, arnum*nznum, c1, v2, c2, v3, v3norm );
}

// Additional LSMR calculation of vectors and vector norm for stop conditions
__global__ void KernelVec3Update_Doublefw( int N, double *hbar, double *h, double *x, double ralpha, double *v, double m1, double m2, double m3, double *hbar_r, double *h_r, double *x_r, double *xnorm )
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int blockSize = blockDim.x;
    int col = bid * blockSize + tid;

    // Declare shared memory array
    extern __shared__ volatile double sdata[];

    // Return this thread is out of input matrix
    if( col >= N )
        return;

    // Calculate vectors
    double hbar_n = h[col] - hbar[col] * m1;
    double x_n    = x[col] + hbar_n * m2;
    h_r[col]     = v[col] * ralpha - h[col] * m3;
    x_r[col]     = x_n;
    hbar_r[col]  = hbar_n;

    // Calculate square sum of result vector components for this group
    sdata[tid] = x_n * x_n; // Every thread calculates square of 1 component
    __syncthreads(); // Make sure all threads are done

    // Divide shared memory array in half and add values in second half to the values in the first half
    // repeat process until size 32
    for( int s = blockSize>>1; s > 32; s >>= 1 )
    {
        if( tid < s )
            sdata[tid] += sdata[tid + s];
        __syncthreads(); // Make sure all threads are done
    }

    // Don't need sychronization for single warp
    if( tid < 32 )
    {
        sdata[tid] += sdata[tid + 32];
        sdata[tid] += sdata[tid + 16];
        sdata[tid] += sdata[tid + 8];
        sdata[tid] += sdata[tid + 4];
        sdata[tid] += sdata[tid + 2];
        sdata[tid] += sdata[tid + 1];
    }

    // Zero thread atomically add this group result to the global squared norm
    if( tid == 0 )
        AtomicAddDouble( xnorm, sdata[0] );
}

void CUDAVec3Update_Doublefw( int block_size, int cnum, double ralpha, double *v, double *x, double *h, double *hbar, double m1, double m2, double m3, double *hbar_r, double *h_r, double *x_r, double *xnorm )
{
    int block_num = cnum / block_size + 1;
    KernelVec3Update_Doublefw<<<block_num, block_size, block_size*sizeof(double)>>>( cnum, hbar, h, x, ralpha, v, m1, m2, m3, hbar_r, h_r, x_r, xnorm );
}

