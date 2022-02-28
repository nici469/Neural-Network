using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;

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
            Console.WriteLine("Beginning testing of MLFileReader...");
            Console.ReadKey(true);

            TestMLFileReader();


        }

        static void TestMLFileReader()
        {
            string fileString = "C:\\Users\\IT-IandE\\Desktop\\TestMLReader.txt";
            string[] lines = File.ReadAllLines(fileString);
            if (lines.Length == 0) throw new Exception("no file was read in the location specified");

            //test CutFileString
            //MLFileReader myReader =new MLFileReader();
            //string[] cutString=myReader.CutFileString(lines,3,6);

            
            //int TNtrue = myReader.CheckDataIntegrity(lines, 8,1, "TimeStep_Start","TimeStep_End",2,9);
            //string[] cutstring = myReader.CutFileString(lines, TNtrue, 8);
            //myReader.CondenseMLArrays(cutstring, TNtrue, 8, 2, 2);

            MLFileReader newReader=new MLFileReader(2,fileString);

            Console.WriteLine("MLReader validated");
        }
    }
}
