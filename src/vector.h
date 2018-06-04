#ifndef ACTUAL_DEVICE_VECTOR
#define ACTUAL_DEVICE_VECTOR
template<class T>
class actual_device_vector {
private:
	T *contents;
	size_t _idx;
	size_t _size;
public:
	__device__ actual_device_vector(size_t n){
		contents = new T[n];
		_size=n;
		_idx=0;
	}
	__device__ actual_device_vector(){
		contents = new T[10];
		_size=10;
		_idx=0;
	}
	__device__ ~actual_device_vector(){
		delete[] contents;
	}
	__device__ void push_back(T x){
		contents[_idx++]=x;
		if(_idx==_size){
			T *temp = new T[_size*2];
			for(size_t i = 0; i < _size; i++) temp[i]=contents[i];
			delete[] contents;
			contents = temp;
			_size *= 2;
		}
	}
	__device__ void erase(size_t n){
		if(n==_idx-1) _idx--;
		else if (_idx==0 || n>_idx) return;
		else contents[n]=contents[--_idx];
		if(_idx<_size/2) {
			T *temp = new T[_size/2];
			for(size_t i = 0; i < _idx; i++) temp[i]=contents[i];
			delete[] contents;
			contents = temp;
			_size /= 2;
		}
	}
	__device__ T& operator[](size_t n){
		return contents[n];
	}
	__device__ int size(){
		return _idx;
	}
};
#endif