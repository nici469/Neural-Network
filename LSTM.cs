using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System;
using System.IO;
using System.Diagnostics;
namespace  Counter_Console

{

    public struct NodeStruct
    {
        public int noOfInputNodes, noOfOutputNodes;
        /// <summary>
        /// specify the number of input nodes, then the number of output nodes
        /// </summary>
        /// <param name="inNodes"></param>
        /// <param name="outNodes"></param>
        public NodeStruct(int inNodes, int outNodes)
        {
            noOfInputNodes = inNodes;
            noOfOutputNodes = outNodes;
        }
    }


    /// <summary>
    /// Description of LSTM. copied from lstm peephole and msade entirely public
    /// </summary>
    public class LSTM_
    {

        public double iODelay, propDelay, errorMargin = 0.05;
        public int cycles = 0;
        
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
        //stores the number of nodes in each layer
        int[] nodes;

        /// <summary>
        /// The Nodestruct specifies the number of input and output nodes, while the elements of hiddenNodes array
        /// specify the number of nodes in each hidden layer
        /// </summary>
        /// <param name="myStruct"></param>
        /// <param name="hiddenNodes"></param>
        public LSTM_(NodeStruct myStruct, int[] hiddenNodes, int tStep)
        {
            //CreateDirectory();
            this.L = hiddenNodes.Length + 1;
            T = tStep;
            //T = 4;
            //T = T;
            lnRate = 0.05 / T;//0.005
            InitialiseVarGradients();//what the hell is this doing here?????????
            InitJgdArrays();

            layerWeightsInitialised = new bool[L];

            nodes = new int[hiddenNodes.Length + 2];
            nodes[0] = myStruct.noOfInputNodes;
            for (int i = 0; i < hiddenNodes.Length; i++)
            {
                nodes[i + 1] = hiddenNodes[i];
            }
            int ndlen = nodes.Length - 1;
            nodes[ndlen] = myStruct.noOfOutputNodes;

            /* //save the number of nodes in each layer sequentially to the file nodes.txt
             StreamWriter sw = new StreamWriter("Nodes.txt");
             using (sw)
             {
                 //the number of input nodes is the first entry
                 sw.WriteLine(myStruct.noOfInputNodes);

                 for (int i = 0; i < hiddenNodes.Length; i++)
                 {
                     sw.WriteLine(hiddenNodes[i]);
                 }
                 sw.WriteLine(myStruct.noOfOutputNodes);
             }*/
        }
        /// <summary>
        /// element-wise multiplicaion.... both inputs must be of the same length>>>TESTED
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
        /// element-wise multiplicaion.... all inputs must be of the same length>>>TESTED
        /// </summary>
        /// <param name="r"></param>
        /// <param name="w"></param>
        /// <returns></returns>
        public double[] elemMul(double[] r, double[] w, double[] n)
        {
            if (r.Length != w.Length || r.Length != n.Length || w.Length != n.Length) { throw new Exception("elemSub: arrays are of different length"); }
            double[] output = new double[r.Length];

            for (int i = 0; i < r.Length; i++)
            {
                output[i] = r[i] * w[i] * n[i];
            }
            return output;
        }
        /// <summary>
        /// element-wise addition.... both inputs must be of the same length...TESTED
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
        /// element-wise addition.... both inputs must be of the same length...TESTED
        /// </summary>
        /// <param name="r"></param>
        /// <param name="w"></param>
        /// <returns></returns>
        public double[] elemAdd(double[] r, double[] w, double[] v, double[] k)
        {
            if (r.Length != w.Length || r.Length != v.Length || r.Length != k.Length) { throw new Exception("elemAdd: arrays are of different length"); }
            double[] output = new double[r.Length];

            for (int i = 0; i < r.Length; i++)
            {
                output[i] = r[i] + w[i] + v[i] + k[i];
            }
            return output;
        }
        /// <summary>
        /// returns the element-wise addition of two matrices...TESTED
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
        /// performs element-wise subtraction r[] -w[]. the two arrays must be of the same length
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
        /// the total number of timesteps...needs to figure out how to initialise this
        /// </summary>
        public int T;
        /// <summary>
        /// used for training in steadily increasing timesteps
        /// </summary>
        //public int T;
        /// <summary>
        /// the stochastic learning rate
        /// </summary>
        public double lnRate;
        double error = 4;
        public void ComputeError()
        {
            int l = L - 1;
            double errMag = 0;
            //double[] error = new double[0];
            for (int t = 0; t < T; t++)
            {
                //error array in each individual timestep
                double[] err1 = elemSub(TargetOutputT[t], HJ[l, t]);
                //double[] err1 = elemSub(Load(".\\TargetOutputT\\TargetOutputT" + t + ".txt"), Load(".\\H\\" + l + "H" + t + ".txt"));

                for (int j = 0; j < err1.Length; j++)
                {
                    errMag += err1[j] * err1[j] / err1.Length;
                }

            }
            errMag = errMag / T;
            if (errMag < error) { error = errMag; }
            else if (errMag > error + 0.001)
            {
                // lnRate *= 0.99;
            }

            if (errMag < errorMargin && (T + 2) <= T) { T += 2; }
            Console.WriteLine("Average error magnitude= {0} ", errMag);
            Console.WriteLine("Incremental timesteps= {0} ", T);
            cycles += 1;
            Console.WriteLine("no of cycles= = {0} ", cycles);

        }

        /// <summary>
        /// performs the forward propagation through the LSTM layers , for all the time steps
        /// </summary>
        public virtual void PropagateForward()
        {

            Stopwatch mywatch = new Stopwatch(); mywatch.Start();

            for (int l = 0; l < L; l++)
            {//propagate through the layers



                ObtainNodes(l);//get the values of D and N
                //Console.WriteLine("d={0}:: n={1}",D,N);
                InitialiseLayerWeights(l);

                if (l == L - 1)
                {
                    for (int t = 0; t < T; t++)
                    {
                        InitialiseAllVariables(l, t);
                        X = ArrayCopy(HJ[l - 1, t]);

                        An = elemAdd(Ba, MatrixMul(Wa, X));
                        A = Tanh(An);

                        H = ArrayCopy(An);//linear
                        HJ[l, t] = H;//save h

                        //if (t == T - 1 && l == L - 1) { ComputeError(); }
                    }
                }

                else
                {

                    for (int t = 0; t < T; t++)
                    {//timesteps


                        InitialiseAllVariables(l, t);
                        try
                        {


                            X = GetInput(l, t);

                            // Console.WriteLine("X was successfully loaded");
                            // for (int b = 0; b < X.Length; b++) { Console.WriteLine("{0} layer X[{1}] tstep{3}= {2}",l,b,X[b],t); }
                        }
                        catch (Exception e) { Console.WriteLine("X: layer " + l + " Timestep" + t + " " + e.Message); }


                        ComputeVariables(l, t);//remove this from comment later


                        //compute layer output array
                        H = elemMul(O, Tanh(C));

                        SaveAllVariables(l, t);

                        //Console.WriteLine("A:{0}, I:{1}, O:{2}, F:{3}", A.Length, I.Length, O.Length, F.Length);
                        // Console.WriteLine("An:{0}, In:{1}, On:{2}, Fn:{3}", An.Length, In.Length, On.Length, Fn.Length);


                    }
                }





            }

            double mycouny = mywatch.ElapsedMilliseconds;
            //Console.WriteLine("total forward prop delay= {0}", mycouny);
            counter += 1;
            if (counter > 10)
            { //ViewOutput();
                counter = 0;
            }
        }


        /// <summary>
        /// saves A,I,O,F,An,In,On,Fn,H,C
        /// it saves in the format layerNo-A-Timestep.txt
        /// </summary>
        public void SaveAllVariables(int layerNo, int timeStep)
        {


            //activations
            AJ[layerNo, timeStep] = A;
            IJ[layerNo, timeStep] = I;
            OJ[layerNo, timeStep] = O;
            FJ[layerNo, timeStep] = F;

            //net inputs
            ANJ[layerNo, timeStep] = An;
            INJ[layerNo, timeStep] = In;
            ONJ[layerNo, timeStep] = On;
            FNJ[layerNo, timeStep] = Fn;
            //output and memory cell
            HJ[layerNo, timeStep] = H;
            CJ[layerNo, timeStep] = C;
            /*
            Save(A, ".\\A\\" + layerNo + "A" + timeStep + ".txt");
            Save(I, ".\\I\\" + layerNo + "I" + timeStep + ".txt");
            Save(O, ".\\O\\" + layerNo + "O" + timeStep + ".txt");
            Save(F, ".\\F\\" + layerNo + "F" + timeStep + ".txt");
            //net inputs
            Save(An, ".\\An\\" + layerNo + "An" + timeStep + ".txt");
            Save(In, ".\\In\\" + layerNo + "In" + timeStep + ".txt");
            Save(On, ".\\On\\" + layerNo + "On" + timeStep + ".txt");
            Save(Fn, ".\\Fn\\" + layerNo + "Fn" + timeStep + ".txt");

            //output and memory cell
            Save(H, ".\\H\\" + layerNo + "H" + timeStep + ".txt");
            Save(C, ".\\C\\" + layerNo + "C" + timeStep + ".txt");

           */

        }
        /// <summary>
        /// computes the value of the memory cell array for the current layer and timestep, which are the inputs
        /// Ct=I.A +F.C(t-1)
        /// </summary>
        /// <param name="timeStep"></param>
        /// <param name="layerNo"></param>
        public void ComputeMemoryCell(int layerNo, int timeStep)
        {
            //Ct=I.A +F.C(t-1)

            C = elemMul(I, A);
            if (timeStep > 0)
            {//remember to alwaya specify the folder of the file to be read
                try
                {
                    //double[] CC = Load(".\\C\\" + layerNo + "C" + (timeStep - 1) + ".txt");//C at t-1 timestep
                    double[] CC = CJ[layerNo, timeStep - 1];//C at t-1 timestep
                    CC = elemMul(CC, F);//multiply with the forget gate activation
                    C = elemAdd(C, CC);//add to the current value C to complete the equation

                }
                catch (Exception e) { Console.WriteLine("ComputeMemoryCell: " + e.Message); }

            }
        }

        /// <summary>
        /// gets the input data required to propagate forward through the particular laye and timestep
        /// l: the layer number
        /// t: the timestep
        /// </summary>
        /// <param name="l"></param>
        /// <param name="t"></param>
        /// <returns></returns>
        public double[] GetInput(int l, int t)
        {

            if (l == 0)
            {//if the input data required is for the first layer
                return BaseT[t];
                //return Load(".\\BaseInput\\BaseInput" + t + ".txt");
            }
            else
            {//else load the output of the previous layer at that particular timestep
                return ArrayCopy(HJ[l - 1, t]);
                //return Load(".\\H\\" + (l - 1) + "H" + t + ".txt");
            }

        }
        
        /// <summary>
        /// initialises the jagged arrays AJ,IJ,OJ,FJ,ANJ.INJ,ONJ,FNJ,CJ,HJ,and the base and TargetOutputT jgd arrays
        /// it also initialises WAJ,WIJ,WOJ,WFJ,BAJ,BIJ,BOJ,BFJ
        /// </summary>
        public void InitJgdArrays()
        {
            AJ = new double[L, T][];
            IJ = new double[L, T][];
            OJ = new double[L, T][];
            FJ = new double[L, T][];

            ANJ = new double[L, T][];
            INJ = new double[L, T][];
            ONJ = new double[L, T][];
            FNJ = new double[L, T][];

            CJ = new double[L, T][];
            HJ = new double[L, T][];

            //the variable gradients
            DAJ = new double[L, T][];
            DIJ = new double[L, T][];
            DOJ = new double[L, T][];
            DFJ = new double[L, T][];

            DANJ = new double[L, T][];
            DINJ = new double[L, T][];
            DONJ = new double[L, T][];
            DFNJ = new double[L, T][];

            DCJ = new double[L, T][];
            DHJ = new double[L, T][];
            DXJ = new double[L, T][];
            BaseT = new double[T][];//the base input
            //BaseT = LoadData(".\\BaseInput\\BaseInput.txt");
            //TargetOutputT = LoadData(".\\TargetOutputT\\TargetOutputT.txt");
            TargetOutputT = new double[T][];//the base input

            //the weights
            WAJ = new double[L][,];
            WIJ = new double[L][,];
            WOJ = new double[L][,];
            WFJ = new double[L][,];

            //the recurrent weights
            UAJ = new double[L][,];
            UIJ = new double[L][,];
            UOJ = new double[L][,];
            UFJ = new double[L][,];

            BAJ = new double[L][];
            BIJ = new double[L][];
            BOJ = new double[L][];
            BFJ = new double[L][];

            //the peephole jagged arrays
            PIJ = new double[L][];
            POJ = new double[L][];
            PFJ = new double[L][];


        }
        /// <summary>
        /// loads a single layer jagged array
        /// </summary>
        /// <param name="filename"></param>
        /// <returns></returns>
        public double[][] LoadData(string filename)
        {
            double[][] output = new double[T][];
            StreamReader sr = new StreamReader(filename);
            using (sr)
            {
                int tStep = int.Parse(sr.ReadLine());//the total number of timesteps is always the first line

                output = new double[tStep][];

                int D = int.Parse(sr.ReadLine());//length of each timestep input is the next line
                string noNeed = sr.ReadLine();//this is usually an empty line

                for (int t = 0; t < tStep; t++)
                {
                    double[] singleStpArray = new double[D];
                    for (int d = 0; d < D; d++)
                    {
                        singleStpArray[d] = double.Parse(sr.ReadLine());
                    }
                    output[t] = singleStpArray;
                    string useless = sr.ReadLine();//an empty line
                }
            }

            return output;
        }
        public double[][] BaseT, TargetOutputT;
        //the weights jagged arrays
        public double[][,] WAJ, WIJ, WOJ, WFJ, UAJ, UIJ, UOJ, UFJ;
        public double[][] BAJ, BIJ, BOJ, BFJ;

        public double[,][] AJ, IJ, OJ, FJ, ANJ, INJ, ONJ, FNJ, CJ, HJ;

        //the variable gradients
        public double[,][] DAJ, DIJ, DOJ, DFJ, DANJ, DINJ, DONJ, DFNJ, DCJ, DHJ, DXJ;
        public double[] A, I, O, F, An, In, Fn, On, C, X, H;
        public double[] Ba, Bi, Bo, Bf;//the bias weights

        public double[] Pi, Po, Pf;//the peephole weights
        public double[][] PIJ, POJ, PFJ;//peephole dtat jagged arrays
        /// <summary>
        /// resets the variables A,I,O,F,An,In,On,Fn,H,X,C to zero length vectors
        /// </summary>
        /// <param name="l"></param>
        /// <param name="t"></param>
        public void InitialiseAllVariables(int l, int t)
        {
            A = new double[0];
            I = new double[0];
            O = new double[0];
            F = new double[0];

            An = new double[0];
            In = new double[0];
            On = new double[0];
            Fn = new double[0];
            H = new double[0];
            X = new double[0];
            C = new double[0];
            //is this really necessary
        }
        /// <summary>
        /// computes An,In,On,and Fn, which are required to obtan A,I,O,F.
        /// also calls computeActivations and ComputeCellMemory
        /// </summary>
        /// <param name="layerNo"></param>
        /// <param name="timeStp"></param>
        public void ComputeVariables(int layerNo, int timeStp)
        {
            ///////////A
            An = MatrixMul(Wa, X, D); An = elemAdd(An, Ba);
            if (timeStp > 0)
            {
                An = elemAdd(An, MatrixMul(Ua, HJ[layerNo, timeStp - 1], D));
            }
            A = Tanh(An);


            ///////////I
            In = MatrixMul(Wi, X, D); In = elemAdd(In, Bi);
            if (timeStp > 0)
            {
                In = elemAdd(In, MatrixMul(Ui, HJ[layerNo, timeStp - 1], D));

                double[] pp = elemMul(Pi, CJ[layerNo, timeStp - 1]);
                In = elemAdd(In, pp);//add peephole contribution
            }
            I = Sigmoid(In);



            //////////////F
            Fn = MatrixMul(Wf, X, D); Fn = elemAdd(Fn, Bf);
            if (timeStp > 0)
            {
                Fn = elemAdd(Fn, MatrixMul(Uf, HJ[layerNo, timeStp - 1], D));

                double[] pp = elemMul(Pf, CJ[layerNo, timeStp - 1]);
                Fn = elemAdd(Fn, pp);//add peephole contribution
            }
            F = Sigmoid(Fn);



            ComputeMemoryCell(layerNo, timeStp);


            ///////O
            On = MatrixMul(Wo, X, D); On = elemAdd(On, Bo);
            if (timeStp > 0)
            {
                On = elemAdd(On, MatrixMul(Uo, HJ[layerNo, timeStp - 1], D));
            }
            double[] ppo = elemMul(Po, C);//peephole contribution to output gate
            On = elemAdd(On, ppo);
            O = Sigmoid(On);





        }
        /// <summary>
        /// saves the program output to a file
        /// </summary>
        public void ViewOutput()
        {
            StreamWriter sw = new StreamWriter(".\\H\\H.txt");
            //StreamWriter sw = new StreamWriter(".\\BaseInput\\BaseInput2.txt");
            using (sw)
            {
                sw.WriteLine(T);

                int mm = HJ[L - 1, 0].Length;
                //int mm = BaseT[0].Length;
                //int mm = TargetOutputT[0].Length;
                sw.WriteLine(mm);

                sw.WriteLine("");
                for (int t = 0; t < T; t++)
                //for (int t = 0; t < T - 2; t++)//t<T-2 so that no conflict occurs, as it may be called right after computeerror
                {
                    for (int nn = 0; nn < mm; nn++)
                    {
                        sw.WriteLine(HJ[L - 1, t][nn]);
                        //sw.WriteLine(BaseT[ t][nn]);
                        //sw.WriteLine(TargetOutputT[t][nn]);
                    }
                    sw.WriteLine("");
                }
            }
        }
        /// <summary>
        /// No longer in use
        /// </summary>
        public void ComputeActivations()
        {
            A = Tanh(An);
            O = Sigmoid(On);
            I = Sigmoid(In);
            F = Sigmoid(Fn);
        }
        /// <summary>
        /// performs the matrix multiplication between the weight and the specified variable
        /// weight-ddXnn... varData-nn array, output-dd array...TESTED
        /// </summary>
        /// <param name="weight"></param>
        /// <param name="varData"></param>
        /// <returns></returns>
        public double[] MatrixMul(double[,] weight, double[] varData, int dd)
        {
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
        /// performs the matrix multiplication between the weight and the specified variable
        /// weight-ddXnn... varData-nn array, output-dd array...TESTED
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
        /// obtains the no of input nodes N, and the number of output nodes D for the specified layer
        /// from the file Nodes.txt
        /// </summary>
        /// <param name="l"></param>
        public void ObtainNodes(int l)
        {
            //string[] nodes = File.ReadAllLines("Nodes.txt");
            N = nodes[l];
            D = nodes[l + 1];
        }

        public double[,] Wa, Wi, Wo, Wf, Ua, Ui, Uf, Uo, dWa, dWi, dWo, dWf, dUa, dUi, dUo, dUf;
        /// <summary>
        /// initialises the weights of a specified layer.
        /// if the weights have been previously initialisd, it simply loads them from a file
        /// </summary>
        /// <param name="l"></param>
        public void InitialiseLayerWeights(int l)
        {
            if (!IsInitialised(l))
            {//if the weights of the specified layer have not been initialised
                Wa = GenRandomMatrix(D, N);
                Wi = GenRandomMatrix(D, N);
                Wo = GenRandomMatrix(D, N);
                Wf = GenRandomMatrix(D, N);

                //the recurrent weights
                Ua = GenRandomMatrix(D, D);
                Ui = GenRandomMatrix(D, D);
                Uo = GenRandomMatrix(D, D);
                Uf = GenRandomMatrix(D, D);

                //the bias
                Ba = GenRandomMatrix(D);
                Bi = GenRandomMatrix(D);
                Bo = GenRandomMatrix(D);
                Bf = GenRandomMatrix(D);

                //the peephole weights                
                Pi = GenRandomMatrix(D);
                Po = GenRandomMatrix(D);
                Pf = GenRandomMatrix(D);

                SaveInitBool(l);
                //save the weights after creating them
                SaveWeights(l);
                Console.WriteLine("weights successfully generated");
            }
            else
            {
                try
                {
                    Wa = WAJ[l];
                    Wi = WIJ[l];
                    Wo = WOJ[l];
                    Wf = WFJ[l];

                    Ua = UAJ[l];
                    Ui = UIJ[l];
                    Uo = UOJ[l];
                    Uf = UFJ[l];

                    Ba = BAJ[l];
                    Bi = BIJ[l];
                    Bo = BOJ[l];
                    Bf = BFJ[l];

                    Pi = PIJ[l];
                    Po = POJ[l];
                    Pf = PFJ[l];
                    /*
                    Wa = LoadWeights(".\\Wa\\" + l + "Wa.txt");
                    Wi = LoadWeights(".\\Wi\\" + l + "Wi.txt");
                    Wo = LoadWeights(".\\Wo\\" + l + "Wo.txt");
                    Wf = LoadWeights(".\\Wf\\" + l + "Wf.txt");
                    //the recurrent weights
                    Ua = LoadWeights(".\\Ua\\" + l + "Ua.txt");
                    Ui = LoadWeights(".\\Ui\\" + l + "Ui.txt");
                    Uo = LoadWeights(".\\Uo\\" + l + "Uo.txt");
                    Uf = LoadWeights(".\\Uf\\" + l + "Uf.txt");
                    // Console.WriteLine("weights successfully loaded");

                    //biases
                    Ba = Load(".\\Ba\\" + l + "Ba.txt");
                    Bi = Load(".\\Bi\\" + l + "Bi.txt");
                    Bo = Load(".\\Bo\\" + l + "Bo.txt");
                    Bf = Load(".\\Bf\\" + l + "Bf.txt");
                    //Console.WriteLine("Biases successfully loaded: Bias Length= {0}",Ba.Length);

                    // Console.WriteLine("uf00= " + Uf[0, 0]);
                    */

                }

                catch (Exception e)
                {
                    Console.WriteLine(e.Message);
                    Console.WriteLine("for some reason, the weight could not be initialised");

                }


            }
        }

        public int counter = 0;
        public void SaveWeights(int l)
        {
            WAJ[l] = ArrayCopy(Wa);
            WIJ[l] = ArrayCopy(Wi);
            WOJ[l] = ArrayCopy(Wo);
            WFJ[l] = ArrayCopy(Wf);

            UAJ[l] = ArrayCopy(Ua);
            UIJ[l] = ArrayCopy(Ui);
            UOJ[l] = ArrayCopy(Uo);
            UFJ[l] = ArrayCopy(Uf);

            BAJ[l] = ArrayCopy(Ba);
            BIJ[l] = ArrayCopy(Bi);
            BOJ[l] = ArrayCopy(Bo);
            BFJ[l] = ArrayCopy(Bf);

            PIJ[l] = ArrayCopy(Pi);
            POJ[l] = ArrayCopy(Po);
            PFJ[l] = ArrayCopy(Pf);
            /*
            //weights
            Save(Wa, ".\\Wa\\" + l + "Wa.txt", D, N);
            Save(Wi, ".\\Wi\\" + l + "Wi.txt", D, N);
            Save(Wo, ".\\Wo\\" + l + "Wo.txt", D, N);
            Save(Wf, ".\\Wf\\" + l + "Wf.txt", D, N);

            //recurrent weights
            Save(Ua, ".\\Ua\\" + l + "Ua.txt", D, D);
            Save(Ui, ".\\Ui\\" + l + "Ui.txt", D, D);
            Save(Uo, ".\\Uo\\" + l + "Uo.txt", D, D);
            Save(Uf, ".\\Uf\\" + l + "Uf.txt", D, D);

            //biases
            Save(Ba, ".\\Ba\\" + l + "Ba.txt");
            Save(Bi, ".\\Bi\\" + l + "Bi.txt");
            Save(Bo, ".\\Bo\\" + l + "Bo.txt");
            Save(Bf, ".\\Bf\\" + l + "Bf.txt");*/

            //stop the watch

        }
        /// <summary>
        /// saves a weight matrix having the specified no of rows and column
        /// </summary>
        /// <param name="data"></param>
        /// <param name="fileName"></param>
        /// <param name="noOfRow"></param>
        /// <param name="noOfColumn"></param>
        public void Save(double[,] data, string fileName, int noOfRow, int noOfColumn)
        {
            StreamWriter sw = new StreamWriter(fileName);
            using (sw)
            {
                sw.WriteLine(noOfRow);//D
                sw.WriteLine(noOfColumn);//N

                for (int i = 0; i < noOfRow; i++)
                {
                    for (int j = 0; j < noOfColumn; j++)
                    {
                        sw.WriteLine(data[i, j]);
                    }
                }
            }
        }
        /// <summary>
        /// loads weight values from the specified file L+W*+.txt
        /// </summary>
        /// <param name="filename"></param>
        /// <returns></returns>
        public double[,] LoadWeights(string filename)
        {
            Stopwatch mywatch = new Stopwatch();
            mywatch.Start();
            if (!File.Exists(filename))
            {
                throw new Exception(filename + "-weight matrix could not be found");
                //return new double[0,0]
            }
            StreamReader sr = new StreamReader(filename);
            using (sr)
            {
                //the first line contains the no Of output nodes for the layer d, the second contains
                //the number of input  nodes n
                int dd = int.Parse(sr.ReadLine());
                int nn = int.Parse(sr.ReadLine());

                double[,] output = new double[dd, nn];
                for (int i = 0; i < dd; i++)
                {//row
                    for (int j = 0; j < nn; j++)
                    {//column
                        string word = sr.ReadLine();

                        if (word != null) { output[i, j] = double.Parse(word); }
                        else
                        {
                            Console.WriteLine("matrix dimensions does not match file being loaded");
                        }

                    }
                }
                return output;
                mywatch.Stop(); iODelay += mywatch.ElapsedMilliseconds;
            }

        }

        /// <summary>
        /// checks if the weights of an lstm layer has been initialised
        /// </summary>
        /// <param name="l"></param>
        /// <returns></returns>
        public bool IsInitialised(int l)
        {
            return layerWeightsInitialised[l];
            /* if (File.Exists(l + "init.txt"))
             {
                 return true;
             }
             else
             {
                 return false;
             }*/
        }
        /// <summary>
        /// each index represents whether the weights of a particular layer has been initialised
        /// </summary>
        public bool[] layerWeightsInitialised;
        /// <summary>
        /// creates a l-init.txt file to show that the weights of a specified lstm layer have been initialised
        /// </summary>
        /// <param name="l"></param>
        public void SaveInitBool(int l)
        {
            layerWeightsInitialised[l] = true;
            /* StreamWriter sw = new StreamWriter(l + "init.txt");
             using (sw)
             {
                 sw.WriteLine(" ");
             }*/
        }

        Random myRandom = new Random();
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
                 // double sign = myRandom.Next();
                 //if (sign > 0.5) { output[i, j] = myRandom.NextDouble() * 0.5; }
                 //else { output[i, j] = myRandom.NextDouble() * 0.5*(-1); }
                    output[i, j] = myRandom.NextDouble() * 0.5;//change to 0.05//previously 0.2
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
                //output[i] = 0.5;//remove this                                        // output[i] = 0.5;//******************************remove thia
            }
            return output;
        }

        public void Save(double[] data, string fileName)
        {
            StreamWriter sw = new StreamWriter(fileName);
            using (sw)
            {
                //the data length is always the first line
                sw.WriteLine(data.Length);
                for (int i = 0; i < data.Length; i++)
                {
                    sw.WriteLine(data[i]);
                }
            }
        }
        public void Save(int[] data, string fileName)
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

        /// <summary>
        /// computes the sigmoid function i/(1+ exp(-s)) given a real input s
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
        /// computes the sigmoid activation of an array of net inputs
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

        //bool firstWeightGradientObtained = false;
        /// <summary>
        /// performs the backward propagation of errors throught the LSTM layers  , for all time steps
        /// </summary>
        public virtual void PropagateBackward()
        {
            iODelay = 0;
            double backPropDelay = 0;
            Stopwatch backWatch = new Stopwatch(); backWatch.Start();

            for (int l = L - 1; l >= 0; l--)
            {//propagate backwards from the last layer L-1


                ObtainNodes(l);
                InitWeightGradients();

                InitialiseLayerWeights(l);//load the weights so the required weight changes can be added to it

                if (l == L - 1)
                {
                    for (int t = 0; t < T; t++)
                    {
                        InitialiseVarGradients();

                        dH = elemSub(HJ[l, t], TargetOutputT[t]);
                        //double[] dHn = elemMul(dH, SigmPrime(H));

                        double[,] Wat = Transpose(Wa);
                        dX = MatrixMul(Wat, dH);

                        DXJ[l, t] = dX;//save the gradient dx


                    }
                }

                else
                {
                    for (int t = T - 1; t >= 0; t--)
                    {//propagate backwards from timestep T-1



                        InitialiseVarGradients();
                        //


                        LoadVariables(l, t);
                        dH = GetBackPropInput(l, t);
                        dO = elemMul(dH, Tanh(C));//dO=dH*tanh(C)
                        dOn = elemMul(dO, SigmPrime(O));//dOn=dO*O*(1-O)

                        //
                        ComputeCellGradient(l, t);

                        //Console.WriteLine("11");
                        ComputeVariableGradient(l, t);
                        ComputeWeightGradient(l, t);

                        //Console.WriteLine("22");
                        ComputeRecurrentGradient(l, t);

                        SaveVariableGradients(l, t);
                        //Console.WriteLine("33");
                    }
                }



                // AddMomentum(l);//add momentum to the weight gradients
                ComputeNewWeight();

                //SaveWeightGradients(l);//save the gradient with the added momentum
                //SaveWeightGradInit(l);


                SaveWeights(l);


            }
            backWatch.Stop(); backPropDelay += backWatch.ElapsedMilliseconds;
            //Console.WriteLine("backprop IO delay= {0}", iODelay);
            //Console.WriteLine("total backprop delay= {0}", backPropDelay);

            //ComputeError();//uncomment this
        }
        /// <summary>
        /// returns true if the weight gradient of the particular layer has been saved at least once..
        /// it uses the init file as an indicator
        /// </summary>
        /// <param name="l"></param>
        /// <returns></returns>
        public bool WeightGradInit(int l)
        {
            bool output = false;
            //if the weight grad initialisation file exists, return true
            if (File.Exists(l + "gradInit")) { output = true; }
            return output;
        }

        /// <summary>
        /// saves the file whose presence signifies that the weight gradient of that layer has been saved
        /// at least once
        /// </summary>
        /// <param name="l"></param>
        public void SaveWeightGradInit(int l)
        {
            if (!File.Exists(l + "gradInit"))
            {
                StreamWriter kk = new StreamWriter(l + "gradInit");
                kk.WriteLine("");
                kk.Close();
            }
        }
       
        public void SaveWeightGradients(int l)
        {
            //weights
            Save(dWa, ".\\dWa\\" + l + "dWa.txt", D, N);
            Save(dWi, ".\\dWi\\" + l + "dWi.txt", D, N);
            Save(dWo, ".\\dWo\\" + l + "dWo.txt", D, N);
            Save(dWf, ".\\dWf\\" + l + "dWf.txt", D, N);

            //recurrent weights
            Save(dUa, ".\\dUa\\" + l + "dUa.txt", D, D);
            Save(dUi, ".\\dUi\\" + l + "dUi.txt", D, D);
            Save(dUo, ".\\dUo\\" + l + "dUo.txt", D, D);
            Save(dUf, ".\\dUf\\" + l + "dUf.txt", D, D);

            //biases
            Save(dBa, ".\\dBa\\" + l + "dBa.txt");
            Save(dBi, ".\\dBi\\" + l + "dBi.txt");
            Save(dBo, ".\\dBo\\" + l + "dBo.txt");
            Save(dBf, ".\\dBf\\" + l + "dBf.txt");
        }

        /// <summary>
        /// loads the variables At,It,Ot,Ft,Ct
        /// </summary>
        /// <param name="l"></param>
        /// <param name="t"></param>
        public void LoadVariables(int l, int t)
        {
            A = AJ[l, t];
            I = IJ[l, t];
            O = OJ[l, t];
            F = FJ[l, t];

            C = CJ[l, t];
            H = HJ[l, t];

            An = ANJ[l, t];
            /*
            A = Load(".\\A\\" + l + "A" + t + ".txt");//At
            I = Load(".\\I\\" + l + "I" + t + ".txt");//It            
            O = Load(".\\O\\" + l + "O" + t + ".txt");//Ot
            F = Load(".\\F\\" + l + "F" + t + ".txt");//Ft

            C = Load(".\\C\\" + l + "C" + t + ".txt");//Ct
            H = Load(".\\H\\" + l + "H" + t + ".txt");//Ht

            An = Load(".\\An\\" + l + "An" + t + ".txt");//At..necessary in computing dAn
            */
        }

        /// <summary>
        /// computes the gradients dUa,dUi,dUo,dUf
        /// </summary>
        /// <param name="l"></param>
        /// <param name="t"></param>
        public void ComputeRecurrentGradient(int l, int t)
        {
            //if the layer under consideration is the last layer, return operation
            if (t == T - 1) { return; }

            //double[] HH = Load(".\\H\\" + l + "H" + t + ".txt");

            //double[] dA1 = Load(".\\dAn\\" + l + "dAn" + (t + 1) + ".txt");//dAn at time t+1
            double[] dA1 = DANJ[l, t + 1];//dAn at time t+1
            dUa = elemAdd(dUa, OuterProduct(dA1, H));//dUa+=outerprod(dAn(t+1), h)

            //double[] dI1 = Load(".\\dIn\\" + l + "dIn" + (t + 1) + ".txt");//dIn at time t+1
            double[] dI1 = DINJ[l, t + 1];//dIn at time t+1
            dUi = elemAdd(dUi, OuterProduct(dI1, H));

            //double[] dO1 = Load(".\\dOn\\" + l + "dOn" + (t + 1) + ".txt");//dOn at time t+1
            double[] dO1 = DONJ[l, t + 1];//dOn at time t+1
            dUo = elemAdd(dUo, OuterProduct(dO1, H));
            //but dFn is zero at t=0
            //double[] dF1 = Load(".\\dFn\\" + l + "dFn" + (t + 1) + ".txt");//dFn at time t+1
            double[] dF1 = DFNJ[l, t + 1];//dFn at time t+1
            dUf = elemAdd(dUf, OuterProduct(dF1, H));

        }
        /// <summary>
        /// computes the new weights from the learning rate and the weight gradients
        /// </summary>
        /// <param name="l"></param>
        /// <param name="t"></param>
        public void ComputeNewWeight()
        {

            //Wa = elemSub(Wa, CombineLnRate(lnRate, dWa));
            Wa = elemAdd(Wa, CombineLnRate(lnRate, dWa));
            Wi = elemAdd(Wi, CombineLnRate(lnRate, dWi));
            Wo = elemAdd(Wo, CombineLnRate(lnRate, dWo));
            Wf = elemAdd(Wf, CombineLnRate(lnRate, dWf));

            Ua = elemAdd(Ua, CombineLnRate(lnRate, dUa));
            Ui = elemAdd(Ui, CombineLnRate(lnRate, dUi));
            Uo = elemAdd(Uo, CombineLnRate(lnRate, dUo));
            Uf = elemAdd(Uf, CombineLnRate(lnRate, dUf));

            Ba = elemAdd(Ba, CombineLnRate(lnRate, dBa));
            Bi = elemAdd(Bi, CombineLnRate(lnRate, dBi));
            Bo = elemAdd(Bo, CombineLnRate(lnRate, dBo));
            Bf = elemAdd(Bf, CombineLnRate(lnRate, dBf));

            Pi = elemAdd(Pi, CombineLnRate(lnRate, dPi));
            Po = elemAdd(Po, CombineLnRate(lnRate, dPo));
            Pf = elemAdd(Pf, CombineLnRate(lnRate, dPf));

        }
        /// <summary>
        /// multiplies the elements of the weightt array with the negative of learning rate R
        /// </summary>
        /// <param name="R"></param>
        /// <param name="weightt"></param>
        /// <returns></returns>
        public double[,] CombineLnRate(double R, double[,] weightt)
        {
            int[] dim = GetDimension(weightt);
            int row = dim[0];
            int col = dim[1];
            double[,] output = new double[row, col];

            for (int i = 0; i < row; i++)
            {
                for (int j = 0; j < col; j++)
                {
                    output[i, j] = (-1) * R * weightt[i, j];
                }
            }
            return output;

        }

        public double[] CombineLnRate(double R, double[] bias)
        {
            int nn = bias.Length;
            double[] output = new double[nn];

            for (int i = 0; i < nn; i++)
            {
                output[i] = (-1) * R * bias[i];
            }
            return output;

        }

        /// <summary>
        /// saves the gradients dC,dX,dF
        /// </summary>
        /// <param name="l"></param>
        /// <param name="t"></param>
        public void SaveVariableGradients(int l, int t)
        {
            Stopwatch watch = new Stopwatch();
            watch.Start();
            DCJ[l, t] = dC;
            DXJ[l, t] = dX;

            DAJ[l, t] = dA;
            DIJ[l, t] = dI;
            DOJ[l, t] = dO;
            DFJ[l, t] = dF;

            DANJ[l, t] = dAn;
            DINJ[l, t] = dIn;
            DONJ[l, t] = dOn;
            DFNJ[l, t] = dFn;

            /*
                        Save(dC, ".\\dC\\" + l + "dC" + t + ".txt");
                        Save(dX, ".\\dX\\" + l + "dX" + t + ".txt");

                        Save(dA, ".\\dA\\" + l + "dA" + t + ".txt");
                        Save(dI, ".\\dI\\" + l + "dI" + t + ".txt");
                        Save(dO, ".\\dO\\" + l + "dO" + t + ".txt");
                        Save(dF, ".\\dF\\" + l + "dF" + t + ".txt");

                        Save(dAn, ".\\dAn\\" + l + "dAn" + t + ".txt");
                        Save(dIn, ".\\dIn\\" + l + "dIn" + t + ".txt");
                        Save(dOn, ".\\dOn\\" + l + "dOn" + t + ".txt");
                        Save(dFn, ".\\dFn\\" + l + "dFn" + t + ".txt");

                        */
        }
        /// <summary>
        /// returns the outer product of two vectors. output[i,j]= vec1[i]*vec2[j]
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
        /// computes the weight gradients dWa, dWo,dWi,dWf
        /// </summary>
        /// <param name="l"></param>
        /// <param name="t"></param>
        public void ComputeWeightGradient(int l, int t)
        {
            // double[] XX = Load(".\\X\\" + l + "X" + t + ".txt");
            double[] XX = GetInput(l, t);

            double[,] AW = OuterProduct(dAn, XX);
            dWa = elemAdd(dWa, AW);

            double[,] IW = OuterProduct(dIn, XX);
            dWi = elemAdd(dWi, IW);

            double[,] OW = OuterProduct(dOn, XX);
            dWo = elemAdd(dWo, OW);

            //df only exists if t>0
            if (t > 0)
            {
                double[,] FW = OuterProduct(dFn, XX);
                dWf = elemAdd(dWf, FW);

                dBf = elemAdd(dBf, dFn);
            }

            dBa = elemAdd(dBa, dAn);
            dBi = elemAdd(dBi, dIn);
            dBo = elemAdd(dBo, dOn);

            if (t < T - 1)
            {
                double[] pp = elemMul(C, DINJ[l, t + 1]);
                dPi = elemAdd(dPi, pp);

                pp = elemMul(C, DFNJ[l, t + 1]);
                dPf = elemAdd(dPf, pp);
            }
            double[] ppo = elemMul(C, dOn);
            dPo = elemAdd(dPo, ppo);

            //Console.WriteLine("computeWeight gradient finalised; t= " + t);
        }

        /// <summary>
        /// computes dC for the specified layer l and timeStep t
        /// </summary>
        /// <param name="l"></param>
        /// <param name="t"></param>
        public void ComputeCellGradient(int l, int t)
        {

            dC = elemMul(dH, O);
            dC = elemMul(dC, TanhPrime(C));//dc=dh*O*tanhprime(c)

            if (t < T - 1)
            {//if the layer under consideration is not the last layer
                //double[] dCt1 = Load(".\\dC\\" + l + "dC" + (t + 1) + ".txt");//dC at timestep t+1
                //double[] dFt1 = Load(".\\dF\\" + l + "dF" + (t + 1) + ".txt");//dF at timestep t+1

                double[] dCt1 = DCJ[l, t + 1];//dC at timestep t+1
                double[] dFt1 = DFJ[l, t + 1];//dF at timestep t+1
                double[] NN = elemMul(dCt1, dFt1);

                dC = elemAdd(dC, NN);//dC+=dC(t+1)*dF(t+1)
            }
            //peephole contribution
            if (t < T - 1)
            {
                //input gate peephole contribution
                double[] pp = elemMul(Pi, DINJ[l, t + 1]);
                dC = elemAdd(dC, pp);

                pp = elemMul(Pf, DFNJ[l, t + 1]);
                dC = elemAdd(dC, pp);
            }
            double[] ppo = elemMul(Po, dOn);//*****On may not have been saved
            dC = elemAdd(dC, ppo);

        }

        /// <summary>
        /// computes the variable gradents da, di, do ,df,dAn,dIn,dOn,Dfn,dX
        /// </summary>
        /// <param name="l"></param>
        /// <param name="t"></param>
        public void ComputeVariableGradient(int l, int t)
        {
            dI = elemMul(dC, A);//dC*At
            dA = elemMul(dC, I);//dC*It
            //Console.WriteLine("Compute Variable Gradients :Da, Di loaded successfully::l={0}, t={1} ",l,t);

            //double[] CC = Load(".\\C\\" + l + "C" + t + ".txt");
            //dO = elemMul(dH, Tanh(C));//dO=dH*tanh(C)

            if (t > 0)
            {//if the layer being considered is not the first layer

                //double[] CC = Load(".\\C\\" + l + "C" + (t - 1) + ".txt");//C at t-1 timestep
                double[] CC = CJ[l, t - 1];//C at t-1 timestep
                dF = elemMul(dC, CC);//dF=dC*C(t-1)

                //double[] FF = Load(".\\F\\" + l + "F" + t + ".txt");//F
                try { dFn = elemMul(dF, SigmPrime(F)); }
                catch (Exception e) { Console.WriteLine("dFn: " + e.Message); }

            }

            //double[] AA = Load(".\\A\\" + l + "A" + t + ".txt");//At
            dAn = elemMul(dA, TanhPrime(An));

            //double[] II = Load(".\\I\\" + l + "I" + t + ".txt");//It
            dIn = elemMul(dI, SigmPrime(I));//

            //double[] OO = Load(".\\O\\" + l + "O" + t + ".txt");//Ot
            //dOn = elemMul(dO, SigmPrime(O));//dOn=dO*O*(1-O)




            //now to compute dX
            double[] WA = MatrixMul(Transpose(Wa), dAn);
            double[] WI = MatrixMul(Transpose(Wi), dIn);
            double[] WO = MatrixMul(Transpose(Wo), dOn);

            double[] WF = new double[N];
            if (dFn.Length > 0) { WF = MatrixMul(Transpose(Wf), dFn); }


            dX = elemAdd(WA, WI, WO, WF);
            //Console.WriteLine("Compute Variable Gradients :CVG completed l={0}, t={1} ", l, t);
        }
        /// <summary>
        /// returns the element-wise Tanh derivative of an array.. 1-Tanh2(input)
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public double[] TanhPrime(double[] input)
        {
            double[] output = new double[input.Length];
            double[] nTanh = Tanh(input);
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = 1 - nTanh[i] * nTanh[i];
            }
            return output;
        }

        /// <summary>
        /// returns the element-wise derivative of a sigmoid function x(1-x)..TESTED
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
        public double[] dA, dI, dO, dF, dC, dH, dX, dAn, dIn, dOn, dFn;
        /// <summary>
        /// initialises all variable gradients -dA,dI,dO,dF,dC,dH,dX to zero length, so the can be used afresh in a new timestep
        /// </summary>
        public void InitialiseVarGradients()
        {
            dA = new double[0];
            dI = new double[0];
            dO = new double[0];
            dF = new double[0];

            dAn = new double[0];
            dIn = new double[0];
            dOn = new double[0];
            dFn = new double[0];

            dC = new double[0];
            dH = new double[0];
            dX = new double[0];
        }
        public double[] dBa, dBi, dBo, dBf;
        public double[] dPi, dPo, dPf;
        /// <summary>
        /// initialises the weight gradients dWa,dWi,dWo,dWf,dUa,dUi,dUo,dUf ,dBa,dBo,dBi,dBf to zero vectors
        /// </summary>
        public void InitWeightGradients()
        {
            dWa = new double[D, N];
            dWi = new double[D, N];
            dWo = new double[D, N];
            dWf = new double[D, N];

            dUa = new double[D, D];
            dUi = new double[D, D];
            dUo = new double[D, D];
            dUf = new double[D, D];
            //initialise the bias gradients
            dBa = new double[D];
            dBi = new double[D];
            dBo = new double[D];
            dBf = new double[D];

            dPi = new double[D];
            dPo = new double[D];
            dPf = new double[D];
        }
        /// <summary>
        /// returns the gtadient dH to be propagated backwards through the specifed layer l and timestep t
        /// </summary>
        /// <param name="l"></param>
        /// <param name="t"></param>
        /// <returns></returns>
        public double[] GetBackPropInput(int l, int t)
        {
            if (l != L - 1)
            {//if the layer under consideration is not the last layer

                double[] output = DXJ[l + 1, t];//load the gradient propagated from the layer above
                //double[] output = Load(".\\dX\\" + (l + 1) + "dX" + t + ".txt");//load the gradient propagated from the layer above
                if (t < T - 1)
                {//if the timestep under consideration is not the last time step in the forward dierction

                    //double[] calc = MatrixMul(Transpose(Ua), Load(".\\dAn\\" + l + "dAn" + (t + 1) + ".txt"), D);
                    double[] calc = MatrixMul(Transpose(Ua), DANJ[l, t + 1], D);
                    output = elemAdd(output, calc);//output+=UaT*dA(t+1)

                    //calc = MatrixMul(Transpose(Ui), Load(".\\dIn\\" + l + "dIn" + (t + 1) + ".txt"), D);
                    calc = MatrixMul(Transpose(Ui), DINJ[l, t + 1], D);
                    output = elemAdd(output, calc);//outpit+=UiT*dI(t+1)

                    //calc = MatrixMul(Transpose(Uo), Load(".\\dOn\\" + l + "dOn" + (t + 1) + ".txt"), D);
                    calc = MatrixMul(Transpose(Uo), DONJ[l, t + 1], D);
                    output = elemAdd(output, calc);//output+=transpose(Uo)*dO(t+1)

                    // calc = MatrixMul(Transpose(Uf), Load(".\\dFn\\" + l + "dFn" + (t + 1) + ".txt"), D);
                    calc = MatrixMul(Transpose(Uf), DFNJ[l, t + 1], D);
                    output = elemAdd(output, calc);//output+=transpose(Uf)*dF(t+1)
                }
                return output;

            }

            else
            {//if the layer under consideration is the last layer

                //double[] output = elemSub(Load(".\\H\\" + l + "H" + t + ".txt"), Load(".\\TargetOutputT\\TargetOutputT" + t + ".txt"));
                double[] output = elemSub(HJ[l, t], TargetOutputT[t]);

                if (t < T - 1)
                {//if the timestep under consideration is not the last time step in the forward dierction

                    //double[] calc = MatrixMul(Transpose(Ua), Load(".\\dAn\\" + l + "dAn" + (t + 1) + ".txt"), D);
                    double[] calc = MatrixMul(Transpose(Ua), DANJ[l, t + 1], D);
                    output = elemAdd(output, calc);//output+=UaT*dA(t+1)

                    //calc = MatrixMul(Transpose(Ui), Load(".\\dIn\\" + l + "dIn" + (t + 1) + ".txt"), D);
                    calc = MatrixMul(Transpose(Ui), DINJ[l, t + 1], D);
                    output = elemAdd(output, calc);//outpit+=UiT*dI(t+1)

                    //calc = MatrixMul(Transpose(Uo), Load(".\\dOn\\" + l + "dOn" + (t + 1) + ".txt"), D);
                    calc = MatrixMul(Transpose(Uo), DONJ[l, t + 1], D);
                    output = elemAdd(output, calc);//output+=transpose(Uo)*dO(t+1)

                    // calc = MatrixMul(Transpose(Uf), Load(".\\dFn\\" + l + "dFn" + (t + 1) + ".txt"), D);
                    calc = MatrixMul(Transpose(Uf), DFNJ[l, t + 1], D);
                    output = elemAdd(output, calc);//output+=transpose(Uf)*dF(t+1)
                }
                return output;
            }
        }

        /// <summary>
        /// returns the transpose of the specified matrix[,]....tested
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
        /// returns the dimension {row, columns} of the specified matrix[,].....TESTED
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


    }


}
