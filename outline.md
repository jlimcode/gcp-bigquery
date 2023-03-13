
1. Motivation: why do we need to use data warehouses? why would we pick Google’s?
2. BigQuery internals and design

   
   1. What is BigQuery
3. Working with BigQuery

   
   1. Ways of accessing BigQuery: cloud console, command line tool, client library
   2. Cloud Console

      
      1. Create a table (max size 10GB)
      2. Query public datasets (max query output 1TB/month: use LIMIT)

         
         1. Star bigquery-public-data project
   3. Client Library

      
      1. Authentication - generating a key for local
      2. Exporting BigQuery job to Python notebook in Colab
4. Compare BigQuery manipulation speed vs. local manipulation with pandas. Or, for larger-than-memory, compare to dask speed.
5. Further resources
6. Summary/Suggestions


