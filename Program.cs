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

        static void DoSomething3Timeless()
        {
            Console.WriteLine("Beginning NN training: press any key to continue");
            Console.ReadKey(true);

            RELUSampleGenerator myGen = new RELUSampleGenerator();
            myGen.GenerateData(1000);
            int noOfInpuNodes = myGen.BaseT[0].Length;
            int noOfOutputNodes = myGen.TOutput[0].Length;

            RELU brain = new RELU(new int[] { noOfInpuNodes, 10,10,5 ,noOfOutputNodes});
            
            brain.InitBatchArrays(myGen.BaseT, myGen.TOutput);
            double error = brain.BatchTrain();

            while(error > 0.05)
            {
                error = brain.BatchTrain();
                Console.WriteLine("Approximate SGD error: {0}", error);
            }
            Console.WriteLine("training completed");
            Console.ReadKey(true);

            while (true) {
                Console.WriteLine("Beginning Testing Session");
                Console.WriteLine("type input values to test the ML algorithm::");
                //var input = Console.ReadLine();

                double[] testInput = new double[noOfInpuNodes];
                for (int i = 0; i < noOfInpuNodes; i++)
                {
                    testInput[i] = double.Parse(Console.ReadLine());
                }
                double[] testOutput = brain.PropagateForward(testInput);
                Console.WriteLine("The Algorithm outputs are:");
                string outString = "";

                for (int i = 0; i < noOfOutputNodes; i++)
                {
                    outString += testOutput[i].ToString() + " ,";
                }
                Console.WriteLine(outString);
            }
            
            //if(input=="")


        }
        static void DoSomething2()
        {
            ///SampleGenerator myGenerator = new SampleGenerator();
            SampleGeneratorClass myGenerator = new SampleGeneratorClass();
            //SampleGeneratorTimeless[] genArray = new SampleGeneratorTimeless[1000];
            SampleGeneratorClass[] genArray = new SampleGeneratorClass[30];
            int tstep = 20;

            for (int i = 0; i < genArray.Length; i++)
            {
                genArray[i] = new SampleGeneratorClass();
                genArray[i].GenerateData(tstep);
            }

            //myGenerator.GenerateData(100);
            //NodeStruct mystruct = new NodeStruct(1, 1);
            //LSTM_NEW brain = new LSTM_NEW(mystruct, new int[] { 5, 5 }, 100);
            RELU_RNN_NEW brain = new RELU_RNN_NEW(new int[] { genArray[0].BaseT[0].Length,10, 10,10,10, 5, genArray[0].TOutput[0].Length },tstep);
            //brain.T = 10;
            brain.TargetOutputT = myGenerator.TOutput;
            brain.BaseT = myGenerator.BaseT;
            double error = 2.0;
            while (true)
            {
                while (error > 0.1)
                {
                    //reset the genData
                    //for (int i = 0; i < genArray.Length; i++)
                    //{
                    //    genArray[i] = new SampleGeneratorClass();
                    //    genArray[i].GenerateData(100);
                    // }
                    //reset error
                    error = 0;
                    for (int i = 0; i < genArray.Length; i++)
                    {
                        brain.BaseT = genArray[i].BaseT;
                        brain.TargetOutputT = genArray[i].TOutput;
                        brain.PropagateForward();
                        // Console.ReadKey(true);
                        brain.PropagateBackward();
                        // Console.ReadKey(true);
                        error += (brain.ComputeError()) / genArray.Length;

                    }

                    Console.WriteLine("error: " + error);
                   // if (error < 0.1 && brain.T + 2 < 100) brain.T += 2;
                    //Console.ReadKey(true);
                }
                error = 0;
                for (int i = 0; i < genArray.Length; i++)
                {
                    brain.BaseT = genArray[i].BaseT;
                    brain.TargetOutputT = genArray[i].TOutput;
                    brain.PropagateForward();
                    // Console.ReadKey(true);
                    brain.PropagateBackward();
                    // Console.ReadKey(true);
                    error += (brain.ComputeError()) / genArray.Length;

                }

                Console.WriteLine("error: " + error);

                Console.WriteLine("press key to continue");
                string input = Console.ReadLine();
                if (input == "test")
                {
                    brain.PropageForwardSingleTStep();
                }
                else if (input == "learn")
                {
                    brain.lnRate *= 0.5;
                }
                else if (input == "INCT" && brain.T <= 90)
                {
                    brain.T += 3;
                }
                Console.ReadKey(true);
            }
        }

        static void DoSomething()
        {
            ///SampleGenerator myGenerator = new SampleGenerator();
            SampleGeneratorClass myGenerator = new SampleGeneratorClass();
            SampleGeneratorClass[] genArray = new SampleGeneratorClass[30];

            for(int i = 0; i < genArray.Length; i++)
            {
                genArray[i] = new SampleGeneratorClass();
                genArray[i].GenerateData(100);
            }

            //myGenerator.GenerateData(100);
            //NodeStruct mystruct = new NodeStruct(1, 1);
            //LSTM_NEW brain = new LSTM_NEW(mystruct, new int[] { 5, 5 }, 100);
            RELU_RNN_NEW brain = new RELU_RNN_NEW(new int[] { 1, 10,10,5,5, 1 }, 100);
            brain.T = 10;
            brain.TargetOutputT = myGenerator.TOutput;
            brain.BaseT = myGenerator.BaseT;
            double error = 2.0;
            while (true)
            {
                while (error > 0.1)
                {
                    //reset the genData
                    //for (int i = 0; i < genArray.Length; i++)
                    //{
                    //    genArray[i] = new SampleGeneratorClass();
                    //    genArray[i].GenerateData(100);
                   // }
                    //reset error
                    error = 0;
                        for(int i = 0; i < genArray.Length; i++)
                        {
                            brain.BaseT = genArray[i].BaseT;
                            brain.TargetOutputT = genArray[i].TOutput;
                            brain.PropagateForward();
                            // Console.ReadKey(true);
                            brain.PropagateBackward();
                            // Console.ReadKey(true);
                            error += (brain.ComputeError())/genArray.Length;
                            
                        }

                    Console.WriteLine("error: " + error);
                    if (error < 0.1&&brain.T+2<100) brain.T += 2;
                    //Console.ReadKey(true);
                }
                error = 0;
                for (int i = 0; i < genArray.Length; i++)
                {
                    brain.BaseT = genArray[i].BaseT;
                    brain.TargetOutputT = genArray[i].TOutput;
                    brain.PropagateForward();
                    // Console.ReadKey(true);
                    brain.PropagateBackward();
                    // Console.ReadKey(true);
                    error += (brain.ComputeError()) / genArray.Length;

                }

                Console.WriteLine("error: " + error);

                Console.WriteLine("press key to continue");
                string input = Console.ReadLine();
                if (input == "test")
                {
                    brain.PropageForwardSingleTStep();
                }
                else if(input == "learn")
                {
                    brain.lnRate *= 0.5;
                }
                else if (input == "INCT"&&brain.T<=90)
                {
                    brain.T += 3;
                }
                Console.ReadKey(true);
            }
        }
        static void Main(string[] args)
        {
            //DoSomething3Timeless();//tested and works
            DoSomething2();
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
            random = new Random();
            TrainRNN();

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
            //double[][] d = myReader.GetTargetOutput();
            //string[] cutString=myReader.CutFileString(lines,3,6);

            
            //int TNtrue = myReader.CheckDataIntegrity(lines, 8,1, "TimeStep_Start","TimeStep_End",2,9);
            //string[] cutstring = myReader.CutFileString(lines, TNtrue, 8);
            //myReader.CondenseMLArrays(cutstring, TNtrue, 8, 2, 2);

            MLFileReader newReader=new MLFileReader(2,fileString);
            double[][] TestToutput = newReader.GetTargetOutput();
            double[][] TestMLInput = newReader.GetTrainingInput();


            Console.WriteLine("MLReader validated");

            
            
            
        }
        static void TrainRNN()
        {
            int noOfInputNodes = 2;
            int tStep = 100;
            //initialise the layer architecture of the neural network
            int[] nodes = { noOfInputNodes, 15, 15,15,15, 5,1 };
            RELU_RNN_NEW brain = new RELU_RNN_NEW(nodes, tStep);

            Console.WriteLine("beginning RNN training");
            Console.ReadKey(true);
            Console.WriteLine("Type a comand: train or test");
            
            string userInput = Console.ReadLine();
            //continue training as long as the user types "train"
            while (userInput == "train"||userInput=="")
            {
                TrainRNN(5, brain, tStep, noOfInputNodes);
                Console.WriteLine("training session complete");
                Console.WriteLine("type next command");
                brain.lnRate *= 0.9;
                userInput = Console.ReadLine();
            }

            if (userInput == "test")
            {
                brain.PropageForwardSingleTStep();
            }
            
            Console.ReadKey(true);
        }


        static void TrainRNN(int noOfSamples,RELU_RNN_NEW brain, int tStep, int noOfInputNodes)
        {
            //test
            double[][] test = new double[3][];
            Console.WriteLine("the length of test is {0}, and should be 3",test.Length);
            //end jgd length test
            double[][] BaseInput, TOutput;
            BaseInput = GenSingleInputSampleJgd(tStep, noOfInputNodes);
            TOutput = GenTOutput(BaseInput);
           
            for(int i = 0; i < 1; i++)
            {
                //BaseInput = GenSingleInputSampleJgd(tStep, noOfInputNodes);
                //TOutput = GenTOutput(BaseInput);
                brain.BaseT = BaseInput;
                brain.TargetOutputT = TOutput;
                double error=brain.Train();
                Console.WriteLine("Current sample error is: " + error);
                
            }
        }

        static double[][] GenTOutput(double[][] baseInput)
        {
            int T = baseInput.Length;
            double[][] TOutput = new double[T][];
            for(int i = 0; i < T; i++)
            {
                TOutput[i] = ComputeSingleOutputArray(baseInput[i]);
            }
            return TOutput;
        }

        static double count = 0;//our parameter for adding patterns
        static double[] ComputeSingleOutputArray(double[] input)
        {
            double out1;
            foreach(double e in input)
            {
                count += e;
            }
            //count += 0.01;

            //here we add the eciding pattern
            if (count >=5) {//output a 1 once the sum of inputs reach 15
                count = 0;out1 = 5;
            }
            else { out1 = 0; }
            //out1 = Math.Sin(count);
            return new double[] { out1 };
        }


        static double[][] GenSingleInputSampleJgd(int TStep, int noOfInputNodes)
        {
            double[][] output = new double[TStep][];
            for(int i = 0; i < TStep; i++)
            {
                output[i] = GenerateSingleTStepInput(noOfInputNodes);
            }
            return output;
        }
        static Random random;
        static double[] GenerateSingleTStepInput(int noOfInputNodes)
        {
            double[] output = new double[noOfInputNodes];
            for(int i=0;i< noOfInputNodes; i++)
            {
                output[i] = random.NextDouble();//return a random numbr between 0 and 5
            }
            return output;
        }
    }
}
