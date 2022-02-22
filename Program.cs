using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Counter_Console
{
    class Program
    {
        NNInterface n;

        void Run()
        {

        }
        static void Main(string[] args)
        {
            Methods M = new Methods();
            double[] v1 = { 1, 2 };
            Vector V1 = new Vector(v1);
            double[] v2 = { 2, 3 };
            Vector V2 = new Vector(v2);
            Matrix bm = V1 ^ V2;
           
            double[,] v3 = M.OuterProduct(v1, v2);
            //v3 = M.Transpose(v3);
            //v3 = M.elemMul(2, v3);
            Matrix mi = new Matrix(v3);
            v3 = mi.T().data;
            Console.WriteLine("v3[0,0]= {0} :: v3[0,1]= {1}",v3[0,0],v3[0,1]);
            Console.WriteLine("v3[1,0]= {0} :: v3[1,1]= {1}", v3[1, 0], v3[1, 1]);

            Vector vi = new Vector(new double[] { 1, 2 });
            Vector vi2 = new Vector(new double[] { 2, 4 });
            Vector mm =  vi2*3;

           
            

            //v3 = M.elemMul(2, v3);
            double[,] v4 = new double[4, 3];
            int[] dim = M.GetDimension(v4);


            Console.ReadKey(true);

        }
    }
}
