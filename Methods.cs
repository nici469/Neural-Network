using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Counter_Console
{
    /// <summary>
    /// contains all matrix and vector operation methods...M-CHECKED
    /// </summary>
    public class Methods
    {
        /// <summary>
        /// returns the outer product of two vectors. output[i,j]= vec1[i]*vec2[j]...M-Checked
        /// </summary>
        /// <param name="vec1"></param>
        /// <param name="vec2"></param>
        /// <returns></returns>
        public double[,] OuterProduct(double[] vec1, double[] vec2)
        {
            int n = vec1.Length;
            int m = vec2.Length;

            double[,] output = new double[n, m];
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < m; j++)
                {
                    output[i, j] = vec1[i] * vec2[j];
                }
            }
            return output;
        }

        /// <summary>
        /// returns the transpose of the specified matrix[,]....M-CHECKED
        /// </summary>
        /// <param name="matrix"></param>
        /// <returns></returns>
        public double[,] Transpose(double[,] matrix)
        {
            int[] dim = GetDimension(matrix);
            int nRows = dim[0];//the number of rows
            int nCol = dim[1];//the number of columns

            double[,] output = new double[nCol, nRows];
            for (int i = 0; i < nCol; i++)
            {
                for (int j = 0; j < nRows; j++)
                {
                    output[i, j] = matrix[j, i];
                }
            }
            return output;

        }

        /// <summary>
        /// returns the element-wise addition of two matrices...M-TESTED
        /// </summary>
        /// <param name="w1"></param>
        /// <param name="w2"></param>
        /// <returns></returns>
        public double[,] elemAdd(double[,] w1, double[,] w2)
        {
            int[] dim1 = GetDimension(w1);
            int[] dim2 = GetDimension(w2);
            if (dim1[0] != dim2[0] || dim1[1] != dim2[1])
            {
                throw new Exception("ElemAdd-Matrix: Matrix dimensions do not match");
            }

            int row = dim1[0];
            int col = dim1[1];
            double[,] output = new double[row, col];
            for (int i = 0; i < row; i++)
            {
                for (int j = 0; j < col; j++)
                {
                    output[i, j] = w1[i, j] + w2[i, j];
                }
            }
            return output;
        }

        /// <summary>
        /// returns the dimension {row, columns} of the specified matrix[,].....M-TESTED
        /// </summary>
        /// <param name="matrix"></param>
        /// <returns></returns>
        public int[] GetDimension(double[,] matrix)
        {
            //Console.WriteLine("getdimension called");
            int noOfRows = matrix.GetLength(0);
            int noOfColumns = matrix.Length / noOfRows;

            //bool errFound = false;
            /* try {
                 while (true) {
                     int count = noOfRows;
                     double noNeed = matrix[count, 0];
                     noOfRows++;
                 }
             }
             catch (Exception e) { ; Console.WriteLine("GetDimension: "+e.Message); 
             }*/
            //int noOfColumns = matrix.Length / noOfRows;
            return new int[] { noOfRows, noOfColumns };
        }

        /// <summary>
        /// element-wise addition.... both inputs must be of the same length...M-TESTED
        /// </summary>
        /// <param name="r"></param>
        /// <param name="w"></param>
        /// <returns></returns>
        public double[] elemAdd(double[] r, double[] w)
        {
            if (r.Length != w.Length) { throw new Exception("elemAdd: arrays are of different length"); }
            double[] output = new double[r.Length];

            for (int i = 0; i < r.Length; i++)
            {
                output[i] = r[i] + w[i];
            }
            return output;
        }

        /// <summary>
        /// performs element-wise subtraction r[] -w[]. the two arrays must be of the same length...M-Checked
        /// </summary>
        /// <param name="r"></param>
        /// <param name="w"></param>
        /// <returns></returns>
        public double[] elemSub(double[] r, double[] w)
        {
            if (r.Length != w.Length) { throw new Exception("elemSub: arrays are of different length"); }
            double[] output = new double[r.Length];

            for (int i = 0; i < r.Length; i++)
            {
                output[i] = r[i] - w[i];
            }
            return output;
        }

        /// <summary>
        /// element-wise multiplicaion.... both inputs must be of the same length>>>M-TESTED
        /// </summary>
        /// <param name="r"></param>
        /// <param name="w"></param>
        /// <returns></returns>
        public double[] elemMul(double[] r, double[] w)
        {
            if (r.Length != w.Length) { throw new Exception("elemSub: arrays are of different length"); }
            double[] output = new double[r.Length];

            for (int i = 0; i < r.Length; i++)
            {
                output[i] = r[i] * w[i];
            }
            return output;
        }
        /// <summary>
        /// multiplies the elements of a matrix by a specified value....M-CHECKED
        /// </summary>
        /// <param name="w1"></param>
        /// <param name="w2"></param>
        /// <returns></returns>
        public double[,] elemMul(double d1, double[,] w2)
        {
            
            int[] dim2 = GetDimension(w2);
            

            int row = dim2[0];
            int col = dim2[1];
            double[,] output = new double[row, col];
            for (int i = 0; i < row; i++)
            {
                for (int j = 0; j < col; j++)
                {
                    output[i, j] = d1* w2[i, j];
                }
            }
            return output;
        }


        /// <summary>
        /// performs the matrix multiplication between the weight and the specified variable
        /// weight-ddXnn... varData-nn array, output-dd array...M-TESTED
        /// </summary>
        /// <param name="weight"></param>
        /// <param name="varData"></param>
        /// <returns></returns>
        public double[] MatrixMul(double[,] weight, double[] varData)
        {
            int[] dim = GetDimension(weight);
            int dd = dim[0];//the number of rows

            double[] output = new double[dd];
            int nn = varData.Length;
            if (weight.Length / dd != nn) { throw new Exception("MatrixMul: arrays do not match for multiplication"); }

            for (int i = 0; i < dd; i++)
            {//rows
                for (int j = 0; j < nn; j++)
                {//columns
                    output[i] += weight[i, j] * varData[j];
                }
            }

            return output;
        }
        /// <summary>
        /// computes the sigmoid function 1/(1+ exp(-s)) given a real input s...M-CHECKED
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public double Sigmoid(double input)
        {
            //next: complete this sigmoid method
            double sigmoid = 1 / (1 + Math.Exp(-1 * input));
            return sigmoid;
        }

        /// <summary>
        /// computes the sigmoid activation of an array of net inputs....M-CHECKED
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public double[] Sigmoid(double[] input)
        {
            double[] output = new double[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = Sigmoid(input[i]);
            }
            return output;
        }
        /// <summary>
        /// returns a  vector object containing the sigmoid of the input vector data...M-CHECKED
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public Vector Sigmoid(Vector input)
        {
            double[] outData = Sigmoid(input.data);
            return new Vector(outData);
        }
        /// <summary>
        /// returns the element-wise derivative of a sigmoid function x(1-x)..M-TESTED
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public double[] SigmPrime(double[] input)
        {
            double[] output = new double[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = input[i] * (1 - input[i]);
            }
            return output;
        }
        /// <summary>
        /// M-CHECKED
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public Vector SigmPrime(Vector input)
        {
            Vector output;
            output.data = SigmPrime(input.data);
            return output;
        }

        //new.......................................
        /// <summary>
        /// computes the tanh function
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public double Tanh(double input)
        {
            double output = Math.Tanh(input);
            return output;
        }
        /// <summary>
        /// computes the Tanh activation of an array of net inputs
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public double[] Tanh(double[] input)
        {
            double[] output = new double[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = Tanh(input[i]);
            }
            return output;
        }
        public Vector Tanh(Vector input)
        {
            Vector output;
            output.data = Tanh(input.data);
            return output;
        }
        /// <summary>
        /// the input is unactivated neuron sum output
        /// </summary>
        /// <returns></returns>
        public Vector TanhPrime(Vector input)
        {
            Vector output;
            output = 1 - (Tanh(input) * Tanh(input));
            return output;
        }

        /// <summary>
        /// leaky ReLU activation function x--0.3X
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public double[] ReLU(double[] input)
        {
            double[] output = new double[input.Length];
            for(int i = 0; i < input.Length; i++)
            {
                if (input[i] > 0)
                {
                    output[i] = input[i];
                }
                else { output[i] = 0.3*input[i]; }
            }
            return output;
        }

        public Vector ReLU(Vector input)
        {
            Vector output;
            output.data = ReLU(input.data);
            return output;
        }
        /// <summary>
        /// computes the leaky Relu gradient given the summed input to the neuron. it contains basically{0,1}
        /// the input is the unactivated neuron sum
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public double[] ReLU_Prime(double[] input)
        {
            int N = input.Length;
            double[] output = new double[N];
            for(int i=0;i< N; i++)
            {
                if (input[i] > 0) { output[i] = 1; }
                else { output[i] = 0.3; }
            }
            return output; 
        }
        public Vector ReLU_Prime(Vector input)
        {
            Vector output;
            output.data = ReLU_Prime(input.data);
            return output;
        }

    }
}
