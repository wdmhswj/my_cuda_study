#include <iostream>
#include <math.h>


// 函数：将2个数组中的元素相加
void add(int n, float* x, float* y)
{
    for (int i = 0; i < n; i++) {
        y[i] = x[i] + y[i];
    }
}

int main()
{
    int N = 1 << 20; // 1M elements
    float* x = new float[N];
    float* y = new float[N];

    // 初始化数组
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // 调用 add 函数
    add(N, x, y);

    // check 错误
    float maxError = 0.0f;
    for (int i = 0; i < N; ++i)
    {
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    }
    std::cout << "Max error: " << maxError << std::endl;

    // 释放内存
    delete[] x;
    delete[] y;

    return 0;
}