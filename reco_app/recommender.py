import pandas as pd
import numpy as np
import heapq
import pickle

class Recommender(object):
    """ A simple context-aware recommendation system for music tracks based on [1].
    
        The model is a standard Matrix Factorization approach to Collaborative Filtering,
        modified to include track genre and contextual condition into consideration,
        in addition to user-track interactions. The predicted track rating is computed as:
    
            r_hat = V_u * Q_i + mu_i + B_g,c
    
        where:
            V_u   - row of users' factor matrix corresponding to user u
            Q_i   - column of items' factor matrix corresponding to item i
            mu_i  - mean rating of item i
            B_g,c - bias of genre g and contextual condition c
    
        The parameter matrices V,G,B are estimated from the provided data by minimizing squared
        difference between actual ratings and r_hat, with regularization terms. For optimization
        two methods are provided: Stochastic Gradient Descent and ADAM. Given trained model,
        track's rating under given contextual condition can be computed by calling predict().
        Function recommend() provides a set of recommendations by returning N highest-ranked tracks.
    
        [1] Baltrunas, Linas, et al. "Incarmusic: Context-aware music recommendations in a car." 
            E-Commerce and Web Technologies. Springer Berlin Heidelberg, 2011. 89-100.
    """
    
    def __init__(self, data_file, model_file=None, train_epochs=25, method="sgd"):
        """ Arguments:
            
            data_file    : path to ratings/tracks dataset in xlsx format
            model_file   : path to model.pickle file with full model (includes dataset)
            train_epochs : number of epochs to train the new model
            method       : 'sgd' or 'adam'
        """
        
        if model_file is None:
            data = pd.read_excel(data_file, sheet_name=["ContextualRating", "Music Track"])
            self.contexts = ["DrivingStyle", 
                            "landscape",
                            "mood",
                            "naturalphenomena",
                            "RoadType",
                            "sleepiness", 
                            "trafficConditions", 
                            "weather"]

            self.tracks = data["Music Track"]
            ratings = data["ContextualRating"]
            ratings = ratings.join(self.tracks[["id"," category_id"]].set_index("id"), on="ItemID")
            ratings.rename(columns = { k : k.strip() for k in ratings.keys() }, inplace=True)    # fix column names
            self.ratings = ratings[ratings["Rating"] != 0]    # exclude zero ratings
            
            # Store indices of users and items
            user_ids = ratings["UserID"].unique()
            item_ids = ratings["ItemID"].unique()
            self.user_id_map = { user_ids[i] : i for i in range(len(user_ids)) }
            self.item_id_map = { item_ids[i] : i for i in range(len(item_ids)) }
            
            # Dict with all context category names
            self.context_cats = { c: ratings[ratings[c].notnull()][c].unique().tolist() for c in self.contexts }
            
            # Mean items' ratings
            mu_i = self.ratings.groupby("ItemID")["Rating"].mean()
            self.mu_i = { self.item_id_map[item_id] : mu_i[item_id] for item_id in item_ids }
            
            # MF algorithm parameters
            self.n_epochs = train_epochs
            self.lbd_reg = 0.03     # regularization parameter
            self.n_latent = 50      # number of latent factors
            self.n_genres = len(self.tracks[" category_id"].unique())

            # MF algorithm variables
            self.V = None
            self.Q = None
            self.B = None
            
            if method == "adam":
                self.adam()
            else:
                self.sgd()
            
            self.store_model("model.pickle")
        else:
            with open(model_file, "rb") as f:
                self.__dict__ = pickle.load(f)
                print("Model loaded from {0}".format(model_file))
        
    def store_model(self, model_file):
        with open(model_file, "wb") as f:
            pickle.dump(self.__dict__, f)
            print("Model stored in {0}".format(model_file))
            
    def sgd(self, lr = 0.005):
        """ Stochastic Gradient Descent. 
        """
        n_users = len(self.user_id_map)
        n_items = len(self.item_id_map)
        n_latent = self.n_latent
        n_genres = self.n_genres
        n_contexts = sum([len(cc) for cc in self.context_cats.values()])

        V = np.random.randn(n_users, n_latent) / np.sqrt(n_users)
        Q = np.random.randn(n_latent, n_items) / np.sqrt(n_latent)
        B = np.zeros((n_genres, n_contexts))

        for epoch in range(self.n_epochs):
            print("SGD epoch: {0}".format(epoch))
            
            for (u,i,c,g,r) in self.get_all_samples():
                # Compute V_u * Q_i
                prod = np.dot(V[u,:], Q[:,i])
                
                # Compute current error
                err = r - (prod + self.mu_i[i] + B[g,c])
        
                # Make a gradient step
                B[g,c] += lr * (err - self.lbd_reg * B[g,c])
        
                for f in range(n_latent):
                    vuf = V[u,f]
                    qfi = Q[f,i]
                    V[u,f] += lr * (err * qfi - self.lbd_reg * vuf)
                    Q[f,i] += lr * (err * vuf - self.lbd_reg * qfi)
                            
            print("Objective value:", self.get_objective(V, Q, B))
        
        self.V = V
        self.Q = Q
        self.B = B
        
    def adam(self, lr = 0.005, beta1 = 0.9, beta2 = 0.999, eps = 1e-8):
        """ ADAM optimizer. Uses adaptive step size.
            Possibly slower than SGD, but can eventually find better solutions.
        """
        n_users = len(self.user_id_map)
        n_items = len(self.item_id_map)
        n_latent = self.n_latent
        n_genres = self.n_genres
        n_contexts = sum([len(cc) for cc in self.context_cats.values()])

        V = np.random.randn(n_users, n_latent) / np.sqrt(n_users)
        Q = np.random.randn(n_latent, n_items) / np.sqrt(n_latent)
        B = np.zeros((n_genres, n_contexts))
        
        MV = np.zeros((n_users, n_latent))
        MQ = np.zeros((n_latent, n_items))
        MB = np.zeros((n_genres, n_contexts))
        
        VV = np.zeros((n_users, n_latent))
        VQ = np.zeros((n_latent, n_items))
        VB = np.zeros((n_genres, n_contexts))
        
        for epoch in range(1, self.n_epochs+1):
            print("ADAM epoch: {0}".format(epoch))
            
            alpha = lr * np.sqrt(1 - beta2**epoch) / (1 - beta1**epoch)
            
            for (u,i,c,g,r) in self.get_all_samples():                
                # Compute V_u * Q_i
                prod = np.dot(V[u,:], Q[:,i])
                
                # Compute current error
                err = r - (prod + self.mu_i[i] + B[g,c])
                
                # Compute bias-corrected moment estimates of gradient of B
                grad_B = err - self.lbd_reg * B[g,c]
                MB[g,c] = beta1 * MB[g,c] + (1-beta1) * grad_B
                VB[g,c] = beta2 * VB[g,c] + (1-beta2) * grad_B*grad_B
                
                # Make a gradient step for B
                B[g,c] += alpha * MB[g,c] / (np.sqrt(VB[g,c]) + eps)
                
                for f in range(n_latent):
                    vuf = V[u,f]
                    qfi = Q[f,i]
                    
                    # Compute bias-corrected moment estimates of gradient of V
                    grad_V = (err * qfi - self.lbd_reg * vuf)
                    MV[u,f] = beta1 * MV[u,f] + (1-beta1) * grad_V
                    VV[u,f] = beta2 * VV[u,f] + (1-beta2) * grad_V*grad_V
                    
                    # Make a gradient step for V
                    V[u,f] += alpha * MV[u,f] / (np.sqrt(VV[u,f]) + eps)
                    
                    # Compute bias-corrected moment estimates of gradient of V
                    grad_Q = (err * vuf - self.lbd_reg * qfi)
                    MQ[f,i] = beta1 * MQ[f,i] + (1-beta1) * grad_Q
                    VQ[f,i] = beta2 * VQ[f,i] + (1-beta2) * grad_Q*grad_Q
                    
                    # Make a gradient step for Q
                    Q[f,i] += alpha * MQ[f,i] / (np.sqrt(VQ[f,i]) + eps)
                            
            print("Objective value: {0:.5f}".format(self.get_objective(V, Q, B)))
        
        self.V = V
        self.Q = Q
        self.B = B
    
    def get_all_samples(self):
        """ Generator function to iterate over all samples.
        """
        # Iterate over all possible context categories
        for ctx in self.contexts:
            ratings_cat = self.ratings[self.ratings[ctx].notnull()]
            for cat in self.context_cats[ctx]:
                subset = ratings_cat[ratings_cat[ctx] == cat]
                
                # Iterate over all entries in the current context category
                for _, row in subset.iterrows():
                    u = self.user_id_map[row["UserID"]]
                    i = self.item_id_map[row["ItemID"]]
                    c = self.context_cats[ctx].index(cat)
                    g = row["category_id"]-1
                    r = row["Rating"]
                    
                    yield (u,i,c,g,r)
        
    def get_objective(self, V, Q, B):
        """ Evaluates the model's objective function. 
        """
        objective = 0
        for (u,i,c,g,r) in self.get_all_samples():
            # Compute V_u * Q_i
            prod = np.dot(V[u,:], Q[:,i])
        
            # Compute current error
            err = r - (prod + self.mu_i[i] + B[g,c])

            objective += err**2 + self.lbd_reg * (np.linalg.norm(V[u,:])**2 + np.linalg.norm(Q[:,i])**2 + B[g,c]**2)
        
        return objective

    def _predict(self, u, i, g, c):
        est = np.dot(self.V[u,:], self.Q[:,i]) + self.mu_i[i] + self.B[g,c]
        return min(5, max(1, est))
                
    def predict(self, user_id, item_id, ctx, cat):
        """ Computes predicted rating for user_id and item_id, under context ctx category cat.
        """
        known_user = user_id in self.user_id_map
        known_item = item_id in self.item_id_map
        
        if not known_item:
            raise Exception("predict error: invalid track ID")
            
        g = self.tracks[self.tracks["id"] == item_id][" category_id"].values[0]-1
        c = self.context_cats[ctx].index(cat)
        i = self.item_id_map[item_id]
        
        est = self.mu_i[i] + self.B[g,c]
        
        if known_user:
            u = self.user_id_map[user_id]
            est += np.dot(self.V[u,:], self.Q[:,i])
                    
        return min(5, max(1, est))
    
    def recommend(self, user_id, ctx, cat, N=10):
        """ Returns top-N ranked items for user_id, under context ctx category cat.
        """
        predictions = []
        for item_id in self.item_id_map.keys():
            est = self.predict(user_id, item_id, ctx, cat)
            heapq.heappush(predictions, (est, item_id))
            
        return heapq.nlargest(N, predictions)

    def get_accuracy_metrics(self):        
        mse = np.mean([float(r - self._predict(u,i,g,c))**2 for (u,i,c,g,r) in self.get_all_samples()])
        rmse = np.sqrt(mse)
        mae = np.mean([float(abs(r - self._predict(u,i,g,c))) for (u,i,c,g,r) in self.get_all_samples()])
        Rsq = 1 - (mse / self.ratings["Rating"].var())
        
        return { "Root Mean Squared Error" : rmse, "Mean Absolute Error" : mae, "R^2" : Rsq }
        