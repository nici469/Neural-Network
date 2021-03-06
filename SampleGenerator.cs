using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Counter_Console
{
    /// <summary>
    /// This one with RELU_RNN_NEW successfully learnt sinewave
    /// </summary>
    class SampleGenerator
    {
        public SampleGenerator()
        {
            //Directory.CreateDirectory(".\\BaseInput");
            //Directory.CreateDirectory(".\\TargetOutput");

        }
        Random myRandom = new Random();
        double count1, count2, count3;
        double[] GenRandomINput(int vectLength)
        {

            double[] output = new double[vectLength];
            for (int i = 0; i < vectLength; i++)
            {
                double rand = myRandom.NextDouble();
                if (rand > 0.3) { output[i] = 1; }
                else { output[i] = 1; }
                output[i] = 1;
                //output[i] = myRandom.Next(0,2);
            }
            return output;
        }
        /// <summary>
        /// vector must have a length of 3.. checks if the count condition of each output has been reached, and gives a value
        /// of 1 for that output, otherwise ,the value is zero
        /// </summary>
        /// <param name="vector"></param>
        double[] CheckCount(double[] vector)
        {
            double out1 = -1, out2 = 0, out3 = 0;
            count1 += vector[0];
            // count2 += vector[1];
            // count3 += vector[2];

            //if (count1 > 5) {  out1 = 1;  count1=0;  }
            //if (count2 > 4) { out2 = 1; count2 = 0; }
            //if (count3 > 5) { out3 = 1; count3 = 0; }
            //out1 = -1;
            out1 = Math.Sin(count1 *(2 * 3.14)/20);
            // return new int[] { out1,out2,out3};
            return new double[] { out1 };//, out2, out3 };
        }
        public double[][] BaseT;
        public double[][] TOutput;
        public void GenerateData(int timeStep)
        {
            BaseT = new double[timeStep][];
            TOutput = new double[timeStep][];
            for (int t = 0; t < timeStep; t++)
            {
                double[] baseInput = GenRandomINput(1);
                double[] targetOutPut = CheckCount(baseInput);
                BaseT[t] = baseInput;
                TOutput[t] = targetOutPut;

                //Save(baseInput, ".\\BaseInput\\BaseInput" + t + ".txt");
                //Save(targetOutPut, ".\\TargetOutput\\TargetOutput" + t + ".txt");
            }
        }
        void Save(int[] data, string fileName)
        {
            StreamWriter sw = new StreamWriter(fileName);
            using (sw)
            {
                sw.WriteLine(data.Length);
                for (int i = 0; i < data.Length; i++)
                {
                    sw.WriteLine(data[i]);
                }
            }
        }

    }

    class SampleGeneratorTimeless
    {
        public SampleGeneratorTimeless()
        {
            //Directory.CreateDirectory(".\\BaseInput");
            //Directory.CreateDirectory(".\\TargetOutput");

        }
        static Random myRandom = new Random();
        double count1, count2, count3;

        //odd versus even sum classification
        double[] GenRandomINput(int vectLength)
        {

            double[] output = new double[vectLength];
            for (int i = 0; i < vectLength; i++)
            {
                double rand = myRandom.NextDouble();
                
                output[i] = rand*10;
                //output[i] = myRandom.Next(0,2);
            }
            return output;
        }
        /// <summary>
        /// vector must have a length of 3.. checks if the count condition of each output has been reached, and gives a value
        /// of 1 for that output, otherwise ,the value is zero
        /// </summary>
        /// <param name="vector"></param>
        double[] CheckCount(double[] vector)
        {
            double out1 = -1, out2 = 0, out3 = 0;
            double count = 0;
            //get the sum of the vector elements.
            ///simple y=x classifier
            double y = vector[0];
            double x = vector[1];
            if (y > x)
            {
                out1 = -1;
            }
            else
            {
                out1 = 1;
            }
            // count2 += vector[1];
            // count3 += vector[2];

            //if (count1 > 5) {  out1 = 1;  count1=0;  }
            //if (count2 > 4) { out2 = 1; count2 = 0; }
            //if (count3 > 5) { out3 = 1; count3 = 0; }
            //out1 = -1;
            //out1 = Math.Sin(count1 * (2 * 3.14) / 20);
            // return new int[] { out1,out2,out3};
            return new double[] { out1 };//, out2, out3 };
        }
        public double[][] BaseT;
        public double[][] TOutput;
        public void GenerateData(int tstep)
        {
            BaseT = new double[tstep][];
            TOutput = new double[tstep][];
            for (int t = 0; t < tstep; t++)
            {
                double[] baseInput = GenRandomINput(2);
                double[] targetOutPut = CheckCount(baseInput);
                BaseT[t] = baseInput;
                TOutput[t] = targetOutPut;

                //Save(baseInput, ".\\BaseInput\\BaseInput" + t + ".txt");
                //Save(targetOutPut, ".\\TargetOutput\\TargetOutput" + t + ".txt");
            }
        }
        void Save(int[] data, string fileName)
        {
            StreamWriter sw = new StreamWriter(fileName);
            using (sw)
            {
                sw.WriteLine(data.Length);
                for (int i = 0; i < data.Length; i++)
                {
                    sw.WriteLine(data[i]);
                }
            }
        }

    }

    /// <summary>
    /// A class for generating samples to train a non-recurrent (non-time-series) neural network
    /// </summary>
    class RELUSampleGenerator
    {
        public RELUSampleGenerator()
        {
            
        }
        /// <summary>
        /// a random number object
        /// </summary>
        static Random myRandom = new Random();

        

        /// <summary>
        /// returns an array of length vecLength containing random real
        /// numbers between 0 and 10
        /// </summary>
        /// <param name="vectLength"></param>
        /// <returns></returns>
        double[] GenRandomINput(int vectLength)
        {

            double[] output = new double[vectLength];
            for (int i = 0; i < vectLength; i++)
            {
                double rand = myRandom.NextDouble();

                output[i] = rand * 10;
                //output[i] = myRandom.Next(0,2);
            }
            return output;
        }
        
        /// <summary>
        /// for computing any custoom function based on the input array to be used for training 
        /// a non-timeseries (non-recurrent) neural network
        /// </summary>
        /// <param name="vector"></param>
        /// <returns></returns>
        double[] ComputeCustomFunction(double[] vector)
        {
            double out1 = -1, out2 = 0, out3 = 0;
            double count = 0;
            //get the sum of the vector elements.
            ///simple y=x classifier
            double y = vector[0];
            double x = vector[1];
            if (y > x+2)
            {
                out1 = -1;
            }
            else
            {
                out1 = 1;
            }
            
            return new double[] { out1 };//, out2, out3 };
        }
        public double[][] BaseT;
        public double[][] TOutput;

        /// <summary>
        /// Generates S number of non-recurrent Training data samples 
        /// to be stored in the object BaseT[S][] and TOutput[S][] jagged arrays
        /// </summary>
        /// <param name="S"></param>
        public void GenerateData(int S)
        {
            BaseT = new double[S][];
            TOutput = new double[S][];
            for (int s = 0; s < S; s++)
            {
                double[] baseInput = GenRandomINput(2);
                double[] targetOutPut = ComputeCustomFunction(baseInput);
                BaseT[s] = baseInput;
                TOutput[s] = targetOutPut;

            }
        }
        void Save(int[] data, string fileName)
        {
            StreamWriter sw = new StreamWriter(fileName);
            using (sw)
            {
                sw.WriteLine(data.Length);
                for (int i = 0; i < data.Length; i++)
                {
                    sw.WriteLine(data[i]);
                }
            }
        }

    }

}
