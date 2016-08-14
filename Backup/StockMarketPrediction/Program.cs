using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
namespace StockMarketPrediction
{
    class Program
    {

        public static List<StockData> LoadStockMarketData(string filename)
        {
            List<StockData> AllTheData = new List<StockData>();

            StreamReader reader = null;
            try
            {
                reader = new StreamReader(filename);

                reader.ReadLine(); //ignore the first line, as this just states headings.

                do
                {
                    string recordText = reader.ReadLine();
                    StockData record = new StockData();
                    string[] recordTextArray = recordText.Split(',');

                    record.Date = DateTime.Parse(recordTextArray[0]);
                    record.Value = Double.Parse(recordTextArray[1]);

                } while (!reader.EndOfStream);
            }
            catch (Exception ex)
            {

                throw ex;
            }
            finally
            {
                if (reader != null)
                    reader.Close();

            }
            return AllTheData;
        }

        public static Dictionary<DateTime, Double> LoadInterestRateData(string filename)
        {
            Dictionary<DateTime, Double> interestRateData = new Dictionary<DateTime, Double>();

            StreamReader reader = null;
            try
            {
                reader = new StreamReader(filename);

                do
                {
                    string[] A = reader.ReadLine().Split(',');
                    interestRateData.Add(DateTime.Parse(A[0]), double.Parse(A[1]));

                } while (!reader.EndOfStream);



                
            }
            catch (Exception ex)
            {

                throw ex;
            }
            finally
            {
                if (reader != null)
                    reader.Close();

            }
            return interestRateData;
        }


        static void Main(string[] args)
        {
            Console.WriteLine("Creating a 10-7-1 Network");
            NeuralNetwork stockPredictor = new NeuralNetwork(10, 7, 1);


            Console.WriteLine("Please enter filename where data is found");
            string filename = Console.ReadLine();

            List<StockData> stockMarketData = new List<StockData>();
            stockMarketData = (List<StockData>)LoadStockMarketData(filename);

            //clone stock market data (make copy)

            Console.WriteLine("Please enter filename where interest rate is found:");
            string filenameForInterestRate = Console.ReadLine();
            
            Dictionary<DateTime, Double> interestRateData = LoadInterestRateData(filenameForInterestRate);


            for (int i = 0; i < stockMarketData.Count; i++)
            {
                stockMarketData[i].InterestRate = interestRateData[stockMarketData[i].Date];
            }


            









            Console.ReadLine();





        }


      public struct StockData
        {
            public DateTime Date;
            public double Value;
            public double InterestRate;
        }

    }
}
