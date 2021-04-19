# Parallel Matrix Factorization Techniques for Recommender Systems
The goal of this project is to parallelize matrix factorization for recommender systems by using Pymp, MPI4py, Ray, and Pyspark. The project includes two main process: matrix factorization computation and recommend movies to users. 

1. Matrix Factorization Computation: implementes the regularized matrix factorization model and saves it as a file (mf_result.txt), which can then used to perform recommendation. 
      -  Codes: **mf_serial.py**, **mf_pymp.py**, **mf_mpi.py**, **mf_ray.py**
      -  Parameter values we used: 
            - steps : the maximum number of steps to perform the optimization was set to 5000
            - alpha : the learning rate was set to 0.0002
            - beta  : the regularization parameter was set to 0.02
            - k     : hidden latent features was set to 8

2. Recommend Movies to Users: takes in users as a input and recommends top 25 movies out of his/her unrated movies.
      -  Codes: **rec_serial.py**, **rec_pymp.py**, **rec_mpi.py**, **rec_ray.py**, **pyspark_np.py**
      -  Results for users 1 as follows:

                Top 25 movies recommendation for the user 1
                Pet Sematary (1989) 
                Angel at My Table  An (1990) 
                9 (2009) 
                Gone Girl (2014) 
                Nanook of the North (1922) 
                Ballistic: Ecks vs. Sever (2002) 
                Downloaded (2013) 
                Before the Rain (Pred dozhdot) (1994) 
                Divine Intervention (Yadon ilaheyya) (2002) 
                Kingdom  The (Riget) (1994) 
                Amazing Grace (2006) 
                Great Expectations (1998) 
                Martian Child (2007) 
                Waking Ned Devine (a.k.a. Waking Ned) (1998) 
                Vibes (1988) 
                Au Hasard Balthazar (1966) 
                King of Kings (1961) 
                Trials of Henry Kissinger  The (2002) 
                Dead Poets Society (1989) 
                Charlie Wilson's War (2007) 
                Omega Man  The (1971) 
                Full Metal Jacket (1987) 
                Thinner (1996) 
                Doc Hollywood (1991) 
                Killer's Kiss (1955)
                

3. Dataset: Movie Lens dataset, describes 5-star rating. It contains 100,234 ratings from 718 users to 8,927 movies.
      - **ratings.csv** file is the rating file. Each line of this file after the header row represents one rating of one movie by  one user, with following format: userId, movieId, rating, timestamp
      - **movies.csv** contains movies information. Each line of this file after the header row represents one movie, with following format: movieId, title, genres
      
      The following dataset contains 209,171 ratings from 162,541 users to 25,000,095 movies.
      - **ratings2.csv** file is the rating file. Each line of this file after the header row represents one rating of one movie by  one user, with following format: userId, movieId, rating, timestamp
      - **movies2.csv** contains movies information. Each line of this file after the header row represents one movie, with following format: movieId, title, genres

4. Run: 
      - Prerequisite: 
           ```
           pip install pymp
           pip install mpi4py
           pip install ray
           ```   
      - Serial:
           ```
           python mf_serial.py ratings.csv <k> movies.csv <steps>
           python rec_serial.py ratings.csv mf_result.txt movies.csv <# of users>
           ``` 
      - Pymp: 
           ```
           python mf_pymp.py ratings.csv <k> movies.csv <# of threads> <steps>
           python rec_pymp.py ratings.csv mf_result.txt movies.csv <# of users> <# of threads>
           ``` 
      - MPI4py:
          ```
          mpirun -n <# of processes> python mf_mpi.py ratings.csv <k> movies.csv <steps>
          mpirun -n <# of processes> python rec_mpi.py ratings.csv mf_result.txt movies.csv <# of users>
          ```
      - Ray:
          ```
          python mf_ray.py ratings.csv <k> movies.csv <# of workers> <steps>
          python rec_ray.py ratings.csv mf_result.txt movies.csv <# of users> <# of workers>
          ```

5. Evaluation: 
      - accuracy: **eval.py** can be imported in mf codes to split the train/test set and calculate the rmse.
      - time consumed: While running the mf and rec codes, the time cost will be written into txt files at the same time. 
