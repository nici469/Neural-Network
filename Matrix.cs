using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Counter_Console
{
    /// <summary>
    /// handles all objects and operations involving matrices...M_CHECKED
    /// </summary>
    public struct Matrix
    {
        public static Methods M=new Methods();
        public const string type="Matrix";
        /// <summary>
        /// the actual matrix data in rpws and columns
        /// </summary>
        public double[,] data;
        /// <summary>
        /// copies one matrix into another...M_TESTED
        /// </summary>
        /// <returns></returns>
        public double[,] ArrayCopy(double[,] input)
        {
            double[,] output = new double[input.GetLength(0), input.GetLength(1)];
            for (int i = 0; i < input.GetLength(0); i++)
            {
                for (int j = 0; j < input.GetLength(1); j++)
                {
                    output[i, j] = input[i, j];
                }

            }
            return output;
        }

        public Matrix(double[,] matData) {
            //type = "Matrix";
            data = matData;
            data = ArrayCopy(matData);
            M = new Methods();
        }
        public Matrix(int row, int col)
        {
            //type = "Matrix";
            data = new double[row, col];
            M = new Methods();
        }

        /// <summary>
        /// matrix multiplication....M-CHECKED
        /// </summary>
        /// <param name="M1"></param>
        /// <param name="V1"></param>
        /// <returns></returns>
        public static Vector operator *(Matrix M1, Vector V1)
        {
            double[] outData = M.MatrixMul(M1.data, V1.data);
            Vector output=new Vector(outData);
            return output;
            
        }
        /// <summary>
        /// returns a new matrix which is the transpose of the
       /// original without changing the original matrix data....M-CHECKED
        /// </summary>
        /// <returns></returns>
        public Matrix T()
        {
            double[,] outData = M.Transpose(data);
            return new Matrix(outData);
        }
        /// <summary>
        /// element-wise adition of two matrices...M-Checked
        /// </summary>
        /// <param name="m1"></param>
        /// <param name="m2"></param>
        /// <returns></returns>
        public static Matrix operator +(Matrix m1, Matrix m2) {
            Matrix output;
            output.data = M.elemAdd(m1.data, m2.data);
            return output;
        }
        /// <summary>
        /// multilpes the elements of a matrix by a specified value...M-CHECKED
        /// </summary>
        /// <param name="m1"></param>
        /// <param name="m2"></param>
        /// <returns></returns>
        public static Matrix operator *(double d1, Matrix m2)
        {
           // int[] dim = M.GetDimension(m2.data);
            Matrix output;
            output.data = M.elemMul(d1, m2.data);
            return output;
        }
        /// <summary>
        /// multilpes the elements of a matrix by a specified value...M-CHECKED
        /// </summary>
        /// <param name="m1"></param>
        /// <param name="m2"></param>
        /// <returns></returns>
        public static Matrix operator *( Matrix m2, double d1)
        {
            // int[] dim = M.GetDimension(m2.data);
            Matrix output;
            output.data = M.elemMul(d1, m2.data);
            return output;
        }


    }
    /// <summary>
    /// a class for handling vector operations...M-CHECKED
    /// </summary>
    public struct Vector
    {
        public static Methods M=new Methods();
        public const string type="Vector";
        /// <summary>
        /// the atual vector array data
        /// </summary>
        public double[] data;
        /// <summary>
        /// create a new vector obect...M-CHECKED
        /// </summary>
        /// <param name="vecData"></param>
        public Vector(double [] vecData)
        {
            // type = "Vector";
            data = new double[vecData.Length];
            data = ArrayCopy(vecData);
           
            M = new Methods();
        }
        /// <summary>
        /// copies one array into another...M-CHECKED
        /// </summary>
        /// <returns></returns>
        public double[] ArrayCopy(double[] input)
        {
            double[] output = new double[input.Length];
            for (int j = 0; j < input.Length; j++)
            {
                output[j] = input[j];
            }
            return output;
        }

        /// <summary>
        /// creates a new vector object...M_CHECKED
        /// </summary>
        /// <param name="row"></param>
        public Vector(int row)
        {
            //type = "Vector";
            data = new double[row];
            M = new Methods();
            //Vector vv = new Vector(2);
            //Vector d = (1 + vv);
        }
        /// <summary>
        /// element-wise vector addition...M-CHECKED
        /// </summary>
        /// <param name="x1"></param>
        /// <param name="x2"></param>
        /// <returns></returns>
        public static Vector operator +(Vector x1, Vector x2)
        {
            double[] outdata = M.elemAdd(x1.data, x2.data);
            return new Vector(outdata);
        }
        /// <summary>
        /// element-wise vector subtraction //x1-x2...M-CHECKED
        /// </summary>
        /// <param name="x1"></param>
        /// <param name="x2"></param>
        /// <returns></returns>
        public static Vector operator -(Vector x1, Vector x2)
        {
            double[] outdata = M.elemSub(x1.data, x2.data);
            return new Vector(outdata);
        }
        /// <summary>
        /// adds a value to every element of the vector...M-CHECKED
        /// </summary>
        /// <param name="x1"></param>
        /// <param name="x2"></param>
        /// <returns></returns>
        public static Vector operator +(double  d1, Vector x2)
        {

            double[] outdata = new double[x2.data.Length];
            for(int i = 0; i < outdata.Length; i++)
            {
                outdata[i] = d1 + x2.data[i];
            }
            return new Vector(outdata);
        }
        /// <summary>
        /// adds a value to every element of the vector...M-CHECKED
        /// </summary>
        /// <param name="x1"></param>
        /// <param name="x2"></param>
        /// <returns></returns>
        public static Vector operator +( Vector x2, double d1)
        {

            double[] outdata = new double[x2.data.Length];
            for (int i = 0; i < outdata.Length; i++)
            {
                outdata[i] = d1 + x2.data[i];
            }
            return new Vector(outdata);
        }

        /// <summary>
        /// subtract every element of a vector from a double value...M-CHECKED
        /// </summary>
        /// <param name="x1"></param>
        /// <param name="x2"></param>
        /// <returns></returns>
        public static Vector operator -(double d1, Vector x2)
        {

            double[] outdata = new double[x2.data.Length];
            for (int i = 0; i < outdata.Length; i++)
            {
                outdata[i] = d1 - x2.data[i];
            }
            return new Vector(outdata);
        }
        /// <summary>
        /// subtract a value from every element of a vector...M-CHECKED 
        /// </summary>
        /// <param name="x1"></param>
        /// <param name="x2"></param>
        /// <returns></returns>
        public static Vector operator -(Vector x2, double d1)
        {

            double[] outdata = new double[x2.data.Length];
            for (int i = 0; i < outdata.Length; i++)
            {
                outdata[i] = x2.data[i]-d1;
            }
            return new Vector(outdata);
        }

        /// <summary>
        /// element-wise vector multiplication...M-CHECKED
        /// </summary>
        /// <param name="x1"></param>
        /// <param name="x2"></param>
        /// <returns></returns>
        public static Vector operator *(Vector x1, Vector x2)
        {
            double[] outdata = M.elemMul(x1.data, x2.data);
            return new Vector(outdata);
        }
        /// <summary>
        /// multiplies every element of a vector by a value...M-CHECKED
        /// </summary>
        /// <param name="x1"></param>
        /// <param name="x2"></param>
        /// <returns></returns>
        public static Vector operator *(double d1, Vector x2)
        {

            double[] outdata = new double[x2.data.Length];
            for (int i = 0; i < outdata.Length; i++)
            {
                outdata[i] = d1 * x2.data[i];
            }
            return new Vector(outdata);
        }
        /// <summary>
        /// multiplies every element of a vector by a value...M-CHECKED
        /// </summary>
        /// <param name="x1"></param>
        /// <param name="x2"></param>
        /// <returns></returns>
        public static Vector operator *( Vector x2, double d1)
        {

            double[] outdata = new double[x2.data.Length];
            for (int i = 0; i < outdata.Length; i++)
            {
                outdata[i] = d1 * x2.data[i];
            }
            return new Vector(outdata);
        }

        /// <summary>
        /// the outer product of two vectors B[i,j]= xi[i] * x2[j].....M-CHECKED
        /// </summary>
        /// <param name="x1"></param>
        /// <param name="x2"></param>
        /// <returns></returns>
        public static Matrix operator ^(Vector x1, Vector x2)
        {
            double[,] outdata = M.OuterProduct(x1.data, x2.data);
            return new Matrix(outdata);
        }

    }
}
