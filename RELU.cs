using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Counter_Console
{
    class RELU
    {
        public RELU(int[] Nodes, int tStep)
        {
            L = Nodes.Length - 1;
            
            
            lnRate = 0.005;

            InitJgdArrays();
            nodes = Nodes;
            layerWeightsInitialised = new bool[L];
            M = new Methods();
        }

        public virtual void InitJgdArrays()
        {
            //the wweights
            WIJ = new double[L][,];
            //the bias
            BIJ = new double[L][];

            YJ = new double[L][];
            YNJ = new double[L][];
            XJ = new double[L][];

            DYJ = new double[L][];
            DYNJ = new double[L][];
            DXJ = new double[L][];
            
        }

        /// <summary>
        /// to indicate whenther the batch sapmle data InputSamplesS[S][] and TargetOutputS[S][] have been initialised
        /// </summary>
        bool batchDataInitialised = false;
        /// <summary>
        /// used for initialising the InpusSamplesS[S][] and TargetOutputT[S][] arrays
        /// </summary>
        /// <param name="S"></param>
        public virtual void InitBatchArrays(double[][] inputSamples, Double [][] targetSamples)
        {
            //inputamples and targetSamples must contain the same number of samples
            if (inputSamples.Length != targetSamples.Length) { throw new Exception("trainingdata dimensions do not match"); }
            InputSamplesS = inputSamples;
            TargetOutputS = targetSamples;
            batchDataInitialised = true;
        }


        public static Methods M = new Methods();
        /// <summary>
        /// the total number of layers
        /// </summary>
        public int L;//the total number of layers
        /// <summary>
        /// the number of output nodes for a given layer... must be set for each layer
        /// </summary>
        public int D;
        /// <summary>
        /// the number of input nodes for a given layer.... must be set for each layer
        /// </summary>
        public int N;
        /// <summary>
        /// the stochastic learning rate
        /// </summary>
        public double lnRate;
        /// <summary>
        /// each index represents whether the weights of a particular layer has been initialised
        /// </summary>
        public bool[] layerWeightsInitialised;

        /// <summary>
        /// the weight jagged arrays
        /// </summary>
        public double[][,] WIJ;
        /// <summary>
        /// The bias JGD array
        /// </summary>
        public double[][] BIJ;
        /// <summary>
        /// stores the number of nodes in each layer
        /// </summary>
        public int[] nodes;
        //thwe weights and biases
        public Matrix Wi, dWi;
        public Vector Bi, dBi;
        /// <summary>
        /// InputSamples[S] and TargetOutput[S] contain the respective input and training output data arrays
        /// for every sample s less than or equal to S.. this are not yet used for now
        /// </summary>
        /// 
        private double[][] InputSamplesS, TargetOutputS;
        /*/// <summary>
        /// to store the input and target outpur training arrays for the cuurent training sample
        /// </summary>
        private double[] input, target;*/

        /// <summary>
        /// the variable vectors
        /// </summary>
        public Vector Y, Yn, X;
        public Vector dY, dYn, dX;
        /// <summary>
        /// the corresponding variable jagged array [L][]
        /// </summary>
        public double[][] YJ, YNJ, XJ;
        /// <summary>
        /// the corresponding variable gradient jagged array [L][]
        /// </summary>
        public double[][] DYJ, DYNJ, DXJ;
        public Random myRandom = new Random();

        /// <summary>
        /// returns a dd X nn matrix filled with random values in the range 0-0.05
        /// </summary>
        /// <param name="dd"></param>
        /// <param name="nn"></param>
        /// <returns></returns>
        public double[,] GenRandomMatrix(int dd, int nn)
        {
            double[,] output = new double[dd, nn];

            for (int i = 0; i < dd; i++)
            {//rows
                for (int j = 0; j < nn; j++)
                {//columns
                    double sign = myRandom.NextDouble();
                    if (sign > 0.5) { output[i, j] = myRandom.NextDouble() * 0.05; }
                    else { output[i, j] = myRandom.NextDouble() * 0.05 * (-1); }
                    // output[i, j] = myRandom.NextDouble() * 0.05;//change to 0.05//previously 0.2
                    // output[i, j] = 1.0;//******************************remove thia
                }

            }
            return output;

        }

        /// <summary>
        /// returns a single column array of the required size containing values in the range 0-0.05.
        /// this is used for initialising bias weights
        /// </summary>
        /// <param name="length"></param>
        /// <returns></returns>
        public double[] GenRandomMatrix(int length)
        {
            double[] output = new double[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = myRandom.NextDouble() * 0.05;//0.05
                //output[i] = 0.5;//remove this                                       
            }
            return output;
        }

        /// <summary>
        /// copies one matrix into another
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

        /// <summary>
        /// copies one array into another
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
        /// initialises the weights matrices of a specified layer.
        /// if the weights have been previously initialisd, initialises them with random matrices
        /// </summary>
        /// <param name="l"></param>
        public void InitLayerMarices(int l)
        {
            //if the weights of a paarticulaar layer have not been initialised, generate them
            if (!layerWeightsInitialised[l])
            {
                Wi.data = GenRandomMatrix(D, N);
                Bi.data = GenRandomMatrix(D);

                SaveWeights(l);
                layerWeightsInitialised[l] = true;
                Console.WriteLine("weights successfully generated");
            }
            //if they have been initialised, load them from the jagged arrays
            else
            {
                Wi.data = WIJ[l];
                Bi.data = BIJ[l];
            }
        }
        /// <summary>
        /// store the weight data in the appropraite jagged arrays
        /// </summary>
        /// <param name="l"></param>
        public void SaveWeights(int l)
        {
            WIJ[l] = ArrayCopy(Wi.data);
            BIJ[l] = ArrayCopy(Bi.data);
        }


        /// <summary>
        /// performs the forward and backward propagation through the Neural Network
        /// for all the samples stored in InputSamplesS[S][] and TargetOutputS[S][];
        /// </summary>
        public virtual void BatchTrain()
        {
            //get the number of training samples
            int S = InputSamplesS.Length;
            //if the batch data has not been initialised and this method is called, throw n exception

            if (S == 0||batchDataInitialised==false) throw new Exception("Batch data not initialised");
            //InputSamplesS ust match TargetOutputS in length,i.e, number of samples
            if (InputSamplesS.Length != TargetOutputS.Length)
            {
                throw new Exception("Batch Training: Sample input and target output do not match ");
            }
            //loop through all the samples
            for(int i = 0; i < S; i++)
            {
                double[] inputInstance = ArrayCopy( InputSamplesS[i]);
                PropagateForward(inputInstance);
                ComputeError(TargetOutputS[i]);
                double[] targetOutputInstance= ArrayCopy(TargetOutputS[i]);
                PropagateBackward(targetOutputInstance);
            }
        }
        /// <summary>
        /// forward propagation for a single sample instance
        /// </summary>
        public virtual void PropagateForward(double[] input)
        {
            for (int l = 0; l < L; l++)
            {
                ObtainNodes(l);
                InitLayerMarices(l);
                
                InitVariableVectors(l);
                X.data = input;
                Yn = Wi * X + Bi;
                Y = M.ReLU(Yn);
                //ComputeVariables(l);//no longer necessary
                SaveAllVariables(l);
                
            }
        }

        /// <summary>
        /// Backward propagation for a single sample instance
        /// </summary>
        /// <param name="targetOutput"></param>
        public virtual void PropagateBackward(double[] targetOutput)
        {
            for (int l = L - 1; l >= 0; l--)
            {
                ObtainNodes(l);
                InitWeightGradients();
                InitLayerMarices(l);

                
                InitVariableGradient();
                LoadVariables(l);
                dY = GetBackPropInput(l, targetOutput);
                dYn = dY * M.ReLU_Prime(Yn);
                dX = Wi.T() * dYn;
                dWi += dYn ^ X;
                SaveVariableGradient(l);
                
                //compute new weights
                Wi += -1 * lnRate * dWi;
                Bi += -1 * lnRate * dBi;
                SaveWeights(l);
            }
        }

        /// <summary>
        /// saves the variables Y,Ynand X to their corresponding jgd arrays
        /// </summary>
        /// <param name="l"></param>
        /// <param name="t"></param>
        public void SaveVariableGradient(int l)
        {
            DYJ[l] = dY.data;
            DYNJ[l] = dYn.data;
            DXJ[l] = dX.data;

        }

        /// <summary>
        /// computes the backprop gradient for the layer l
        /// if it is the last layer, it computes the difference between the last layer output and the targetOutput
        /// </summary>
        /// <param name="l"></param>
        /// <param name="targetOutput"></param>
        /// <returns></returns>
        public Vector GetBackPropInput(int l, double[] targetOutput)
        {
            Vector Del;
            //get del from the layer above or from the targetOutput
            if (l != L - 1) { Del.data = DXJ[l + 1]; }
            else { Del.data = M.elemSub(YJ[l], targetOutput); }

            

            return Del;
        }

        /// <summary>
        /// loads the vectors Y,Yn and X
        /// </summary>
        /// <param name="l"></param>
        public void LoadVariables(int l)
        {
            Y.data = YJ[l];
            Yn.data = YNJ[l];
            X.data = XJ[l];
        }

        /// <summary>
        /// intialises the matrix data of dWI, dUi, and dBi to zero vectors
        /// </summary>
        public void InitWeightGradients()
        {
            dWi.data = new double[D, N];
            dBi.data = new double[D];
        }

        /// <summary>
        /// initialise all variable gradient vector data to zero length arrays
        /// </summary>
        public void InitVariableGradient()
        {
            dY.data = new double[0];
            dYn.data = new double[0];
            dX.data = new double[0];
        }

        /// <summary>
        /// saves Y,Yn and X to the corresponding jagged arrays
        /// </summary>
        /// <param name="l"></param>
        public void SaveAllVariables(int l)
        {
            YJ[l] = Y.data;
            YNJ[l] = Yn.data;
            XJ[l] = X.data;

        }








        /// <summary>
        /// obtains the no of input nodes N, and the number of output nodes D for the specified layer
        /// 
        /// </summary>
        /// <param name="l"></param>
        public void ObtainNodes(int l)
        {
            //string[] nodes = File.ReadAllLines("Nodes.txt");
            N = nodes[l];
            D = nodes[l + 1];
        }


        /// <summary>
        /// resets all variable vectors to zero length
        /// </summary>
        /// <param name="l"></param>
        /// <param name="t"></param>
        public void InitVariableVectors(int l)
        {
            Y.data = new double[0];
            Yn.data = new double[0];
            X.data = new double[0];

        }


       

    }
}
