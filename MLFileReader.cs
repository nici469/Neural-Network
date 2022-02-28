using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;

namespace Counter_Console
{
    public class MLFileReader
    {
        
        /// <summary>
        /// The number of input variables
        /// </summary>
        private int Ni;

        /// <summary>
        /// The number of output variables
        /// </summary>
        private int No;

        /// <summary>
        /// the full path of the file containing the data to be read for ML training
        /// </summary>
        private string filePath;

        
        /// <summary>
        /// used to store the position of the start signature of the first timestep in the filePath
        /// </summary>
        int startSignPos;

        /// <summary>
        /// used to store the position of the ending signature of the first timestep in the filePath
        /// </summary>
        int firstEndSignPos;

        /// <summary>
        /// a signature string that specifies the start of a new timestep
        /// </summary>
        string startSign;

        /// <summary>
        /// a signature string that specifies the end of a timestep
        /// </summary>
        string endSign;

        /// <summary>
        /// Jagged array containig the ML input data in format [Timestep] [dataId] or
        /// [T][Ni]
        /// </summary>
        private double[][] MLInput;

        /// <summary>
        /// Jagged array to store the ML target output data in the format [Timestep] [dataId]
        /// or [T][No]
        /// </summary>
        private double[][] TOutput;

        /// <summary>
        /// to store the first timestep value in the filePath. (to be demoted from field variableand 
        /// made a method-local variable)
        /// </summary>
        int Tfirst;
        
        /// <summary>
        /// to store the number of continuous unbroken and validated timesteps that progress
        /// smoothly from the first timestep
        /// </summary>
        int TNtrue;

        /// <summary>
        /// the distance between similar positions between timesteps;
        /// or the line distance between the start signatures of two adjacent timestep data in the filePath
        /// </summary>
        int K;


        /// <summary>
        /// set to true if an object of the class is initialised properly through the designated 
        /// constructors. defaults to false otherwise
        /// </summary>
        bool objectInitialised;

        /// <summary>
        /// an empty constructor of the MLFileReader class is not allowed. An object of the class should
        /// be initialised with startSignPosition and filePath
        /// </summary>
        public MLFileReader()
        {
            //re-expose this exception after testing
            //throw new Exception("MLFileReader object cannot be initialised with empty constructor");
        }
        /// <summary>
        /// StartSignPos is the position of the first start-signature in the File;
        /// FilePath is the full path to the file containing the date to use for ML training
        /// </summary>
        /// <param name="signStart"></param>
        /// <param name="filename"></param>
        public MLFileReader(int StartSignPos, string FilePath)
        {
            startSignPos = StartSignPos;
            filePath = FilePath;
            objectInitialised = true;
            startSign = "TimeStep_Start";
            endSign = "TimeStep_End";
            Init();
        }
        

        /// <summary>
        /// initialises relevant position integers, datalengths, its reads the string[] lines from
        /// the filePath as an array, calls CheckDataIntegrity, calls CutFileString to cut out the relevant part 
        /// of the string[] array gotten from filepath, and calls CondenseMLArrays
        /// </summary>
         public void Init() {
            
            //check to ensure the object of the MLFileReader class has been properly initialised
            if (objectInitialised == false) { throw new Exception("MLFileReader object was not properly initialised"); }
            
            //read all the lines of string in the filePath into an array
            string[] lines = File.ReadAllLines(filePath);

            //the first timestep as seen in the file arrangement, is the 4th element of the lines array. 
            //which corresponds to the 3rd index. 
            Tfirst = int .Parse(lines[3]);

            //all timesteps start from 1, and is never zero. if it is zero, throw an exception
            if (Tfirst==0) { 
                throw new Exception("invalid file: the first timestep could not be read"); 
            }

            Ni = int.Parse(lines[0]);//Ni is the first element of the lines array
            No = int.Parse(lines[1]);//Ni is the second element of the lines array
            K = Ni + No + 2 + 2;//Ni+No data plus 2ts+ 2signatures per timestep

            //neither Ni nor No can be zero
            if (No == 0 || Ni == 0){throw new Exception("inavlid file: missing training data"); }

            firstEndSignPos = startSignPos + Ni + No + 2+1;// it is Ni +No +2(Timesteps) +1 away from the startSignature

            //chack to ensure the start and end signatures have been initialised
            if (startSign == null || endSign == null)
            {
                throw new Exception("a timestep signature has not been initialised or is an emptyvstring");
            }
            TNtrue = CheckDataIntegrity(lines,K,Tfirst,startSign,endSign,startSignPos,firstEndSignPos);
            string[] trueFile = CutFileString(lines, TNtrue, K);//paused

            CondenseMLArrays(trueFile,TNtrue,K,Ni,No);

        }
        /// <summary>
        /// saves the validated training data input and target output set to MLInput[][] and TOutput[]
        /// ::inDataLength And outDataLength are the number of training input and output variables captured in each timestep
        /// respectively. They correspond to Ni and No (public fields)
        /// </summary>
        public void CondenseMLArrays(string[] lines, int noOfTimesteps, int timeStepDistance, int inDataLength, int outDataLength) {
            MLInput = new double[noOfTimesteps][];
            TOutput = new double[noOfTimesteps][];

            for(int i = 0; i < noOfTimesteps; i++){//iterate through each timestep
                //the training input data begins two rows after the timestep signature, 
                //which should be the first element of the string[] lines arraay
                int inDataStart =i*timeStepDistance+2;
                int outDataStart =i*timeStepDistance+2+inDataLength;

                MLInput[i] = new double[inDataLength];//initialise the training input array for the current timestep
                for(int n = 0; n < inDataLength; n++){//iterate through the elements
                    MLInput[i][n] = double.Parse(lines[inDataStart + n]);
                }

                TOutput[i] = new double[outDataLength];//initialise the target output array for the current timestep
                for (int n = 0; n < outDataLength; n++){//iterate through the elements
                    TOutput[i][n] = double.Parse(lines[outDataStart + n]);
                }

            }
           
        }

        /// <summary>
        /// /// checks the integrity of the stringLines[] data array. 
        /// it returns the longest continuous and valid number of timesteps from that starts from the first timestep.
        /// this method is not yet initialised. StartSign is the starting signature, StartSignPos is the position of the first start sign
        /// </summary>
        /// <param name="stringLines"></param>
        /// <param name="timeStepDistance"></param>
        /// <param name="StartSign"></param>
        /// <param name="EndSign"></param>
        /// <param name="StartSignPos"></param>
        /// <param name="FirstEndSignPos"></param>
        /// <returns></returns>
        public int CheckDataIntegrity(string[] stringLines, int timeStepDistance,int firstTimestep, string StartSign, string EndSign,int StartSignPos,int FirstEndSignPos) {
            /// <summary>
            /// to store the estimated number of timesteps in the file
            /// </summary>
            int Ttest;
            int noOfLines = stringLines.Length;
            //for the data to be valid, the number of lines must at least be more than 2(Ni, No)+ 4(2 signs, 2Ts) + 2(1-input and 1 targetv output)
            if (noOfLines < 8)
            {
                throw new Exception("invalid or incomplete data: data length too small");
            }
            //The line just before the last line should contain the last timestep vale in the file
            int Tfile = int.Parse(stringLines[noOfLines - 2]);
            Ttest = (noOfLines - 2) / timeStepDistance;//the estimated- number of timesteps in the file

            //if the estimated number o timesteps in the file is not equal to the value of the supposed last timestep in the file
            //it is either the file is broken or it does not start from timestep 1
            if (Ttest != Tfile)
            {
                Console.WriteLine("File is broken or does not start from timestep 1: Ttest!=Tfile");
            }

            //chack to ensure the start and end signatures have been initialised
            if (StartSign == null || EndSign == null){throw new Exception("a timestep signature has not been initialised or is an empty string");  }
            
            //check to ensure the first starting and ending signature positions are initialised
            if (StartSignPos == 0 || FirstEndSignPos == 0)
            {
                throw new Exception("either the position of the first start sign or the first end sign has not been initialised");
            }

            //the validated number of timesteps, or the number of continuous validated timesteps 
            int Tv = 0;


            //iterate through the data in the file; if the file has no errors, this would be iterating through the timesteps
            //in the file
            for (int i = 0; i < Ttest; i++)
            {   
                //Read the timestep value one line below the starting signature
                int timeStpValue = int.Parse(stringLines[StartSignPos + 1 + i * timeStepDistance]);
                //Read the repeated timestep value one line above the ending signature of each timestep
                int repeatedTstep = int.Parse(stringLines[FirstEndSignPos - 1 + i * timeStepDistance]);

                //check for presence of the correct starting signature in each timestep
                if (stringLines[StartSignPos + i * timeStepDistance] != StartSign )
                {
                    //throw new Exception("unrecognised starting signature");
                    return Tv;
                    break;
                    //return Tv, throw warning or break
                }

                //check for presence of the correct ending signature in each timestep
                else if (stringLines[FirstEndSignPos + i * timeStepDistance] != EndSign)
                {
                    //throw new Exception("unrecognised ending signature");
                    return Tv;
                    break;
                    //return Tv, throw warning or break
                }

                //Note: all timesteps must have a value >=1. a timestep variable value can only be zero if not initialised
                //if Tfirst has not been read or the timestep value read in each iteration after the 
                //startSignature does not follow properly from Tfirst, then
                else if ((firstTimestep==0)||timeStpValue!= (i+firstTimestep))
                {
                    //throw new Exception("break or invalid point in timestep continuity detected");
                    return Tv;
                    break;
                    //return Tv, throw warning or break
                }

                //check to the timestep value just before the ending signature: the timestep value is writtn twice for each timestep
                else if (repeatedTstep != i + firstTimestep)
                {
                    //throw new Exception("break or invalid point in timestep continuity detected");
                    return Tv;
                    break;
                    //return Tv, throw warning or break
                }
                //otherwise, if all checks went well, we have found a valid timestep
                else
                {
                    Tv++;
                }


                

            }
            //return the number of valid and continuous timesteps that follow incrementally from the first timestep in the file
            return Tv;

           
        }


        /// <summary>
        /// this method is not yet initialised.
        /// it custs out a specified number of timesteps from a string array of file lines
        /// the lines containing the values of Ni and No will no longer be included
        /// </summary>
        /// <param name="lineData"></param>
        /// <param name="noOfTimesteps"></param>
        /// <param name="timeStepDistance"></param>
        /// <returns></returns>
        public string[] CutFileString(string[] lineData, int noOfTimesteps, int timeStepDistance)
        {
            string[] outputString = new string[timeStepDistance*noOfTimesteps];//the output will not contain Ni and No again
            
            //iterate through timesteps
            for (int t = 0; t < noOfTimesteps; t++)
            {
                //iterate wuthin the timestep
                for (int j = 0; j < timeStepDistance; j++)
                {
                    outputString[t * timeStepDistance + j] = lineData[t * timeStepDistance + j + 2];//the "2" represents Ni and No that are nolonger included

                }
            }
            return outputString;
        }

        /// <summary>
        /// used to safely return training Input data for an ML training algorithm. it 
        /// returns MLInput[T][Ni]
        /// </summary>
        /// <returns></returns>
        public double [][] GetTrainingInput()
        {
            double[][] output = ArrayCopy(MLInput);
            return output;
        }

        /// <summary>
        /// used to safely return Target output data for an ML training algorithm.
        /// it returns TOutput[T][No]. They are  called externally
        /// </summary>
        /// <returns></returns>
        public double[][] GetTargetOutput()
        {
            double[][] output = ArrayCopy(TOutput);
            return output;
        }

        /// <summary>
        /// for safely replicating a jagged array of dimension [][]. called externally
        /// </summary>
        /// <param name="arrayInput"></param>
        /// <returns></returns>
        double[][] ArrayCopy(double[][] arrayInput)
        {
            //get the number of children array in the parent jagged array
            int noOFChildArray = arrayInput.Length;
            
            double[][] outputArray = new double[noOFChildArray][];

            for(int i = 0; i < noOFChildArray; i++) {//iterate through all the child arrays in the jagged array
                //get the number of element in each child array
                int noOfElements = arrayInput[i].Length;
                          
                outputArray[i] = new double[noOfElements];
                for(int j = 0; j < noOfElements; j++)
                {
                    outputArray[i][j] = arrayInput[i][j];
                }

            }

            return outputArray;
        }






    }
}
