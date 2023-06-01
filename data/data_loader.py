import os
import math
import numpy as np
import pandas as pd
import ipdb


class DataReader(object):
    '''
    A general data reader for different KT datasets.
    This contains the basic data features and aggregates the features according to different learners.
    For specific data features, it will be defined at get_feed_dict function in each KT model class.
    '''
    def __init__(
        self, 
        args, 
        logs
    ):
        '''
        Args:
            prefix:    data folder path
            dataset:   the name of KT dataset
            sep:       the delimiter when loading a csv file
            k_fold:    number of k folder to do cross-validation
            max_step:  the maximum step considered during training; NOTE: sometimes it has also been defined during the pre-processing process
            logs:      the log instance where defining the saving/loading information
        '''
        self.data_dir = args.data_dir
        self.dataset = args.dataset
        self.k_fold = args.kfold
        self.max_step = int(args.max_step)
        self.logs = logs
        
        self.inter_df = pd.read_csv(os.path.join(self.data_dir, self.dataset, 'interactions_{}.csv'.format(args.max_step)), sep='\t')
        if 'problem_id' not in self.inter_df.columns:
            self.inter_df['problem_id'] = self.inter_df['skill_id']

        #  load the ground-truth graph if available  
        self.adj = self.load_ground_truth_graph()
        
        logs.write_to_log_file('Reading data from \"{}\", dataset = \"{}\" '.format(self.data_dir, self.dataset))

        # Aggregate by user
        user_wise_dict = dict()
        cnt, n_inters = 0, 0
        self.data_df = {'train': pd.DataFrame(), 'val': pd.DataFrame(), 'test': pd.DataFrame()}
        
        for user, user_df in self.inter_df.groupby('user_id'):
            user_df = user_df.sort_values('timestamp', ascending=True)
            
            df = user_df[:self.max_step]  

            # TODO current only work with binary correct
            df = df.groupby('skill_id').apply(lambda x: x.assign(
                num_history=np.arange(len(x)),
                num_success=x['correct'].cumsum(),
            ))
            df['num_success'] = np.maximum(df['num_success']-1,0)
            df['num_failure'] = df['num_history'] - df['num_success']
            
            # normalize time stamp -> because we care about the relative time
            new_df = df.sort_values('timestamp', ascending=True)
            new_df['timestamp'] = new_df['timestamp'] - min(new_df['timestamp'])
            
            user_wise_dict[cnt] = {
                'user_id': user,                                                    # the ID of the learner
                'skill_seq': new_df['skill_id'].values.tolist(),                    # the sequence of ID of the skills
                'correct_seq': [round(x) for x in df['correct']],                   # the sequence of the performance corresponding to the skill (binary)
                'time_seq': new_df['timestamp'].values.tolist(),                    # the sequence of the time stamps; it should be in an ascending order
                'problem_seq': new_df['problem_id'].values.tolist(),                # the sequence of ID of the problems; NOTE: one skill can have multiple problems
                'num_history': new_df['num_history'].values.astype(int).tolist(),   # until the time step, the amount of interactions of this specific skill for this learner
                'num_success': new_df['num_success'].values.astype(int).tolist(),   # the amount of interactions where the learner succeeded
                'num_failure': new_df['num_failure'].values.astype(int).tolist(),   # the amount of interactions where the learner failed
            }
            
            cnt += 1
            n_inters += len(df)
        
        self.user_seq_df = pd.DataFrame.from_dict(user_wise_dict, orient='index')
        self.n_users = max(self.inter_df['user_id']) + 1
        self.n_skills = max(self.inter_df['skill_id']) + 1
        self.n_problems = max(self.inter_df['problem_id']) + 1

        self.logs.write_to_log_file('"n_users": {}, "n_skills": {}, "n_problems": {}, "n_interactions": {}'.format(
            self.n_users, self.n_skills, self.n_problems, n_inters
        ))


    def load_ground_truth_graph(self):
        """
        Load the ground truth graph if available, otherwise create an empty graph.

        Returns:
        - adj (ndarray): Adjacency matrix representing the ground truth graph.
        """
        graph_path = os.path.join(self.data_dir, self.dataset, 'adj.npy')
        if os.path.exists(graph_path):
            adj = np.load(graph_path)
        else:
            adj = np.zeros((self.n_skills, self.n_skills))
        return adj


    def gen_fold_data(
        self, 
        k: int
    ):
        '''
        TODO: this function is not used in the current version
        Args:
            k: select the k-th fold to run
        '''
        assert k < self.k_fold
        n_examples = len(self.user_seq_df)
        fold_size = math.ceil(n_examples / self.k_fold)
        
        fold_begin = k * fold_size
        fold_end = min((k + 1) * fold_size, n_examples)
        self.data_df['test'] = self.user_seq_df.iloc[fold_begin:fold_end]
        
        residual_df = pd.concat([self.user_seq_df.iloc[0:fold_begin], self.user_seq_df.iloc[fold_end:n_examples]])
        val_size = int(0.1 * len(residual_df)) # 
        val_indices = np.random.choice(residual_df.index, val_size, replace=False)  # random
        self.data_df['val'] = self.user_seq_df.iloc[val_indices]
        
        self.data_df['train'] = residual_df.drop(val_indices)
        self.data_df['whole'] = self.user_seq_df


    def gen_time_split_data(
        self, 
        train_ratio: float,
        test_ratio: float, 
        val_ratio: float=0.2,
    ):
        '''
        Split the train/test/val based on time steps. 
        self.user_seq_df.keys(): ['user_id', 'skill_seq', 'correct_seq', 'time_seq', 'problem_seq',
                                  'num_history', 'num_success', 'num_failure']
        Args:
            
        '''
        assert train_ratio + test_ratio + val_ratio <= 1

        n_time_steps = len(self.user_seq_df['time_seq'][0])
        self.data_df = {
            'train': dict(),
            'val': dict(),
            'test': dict(),
            'whole': dict(),
        }
        
        train_size = math.ceil(n_time_steps * train_ratio)
        test_size = math.ceil(n_time_steps * test_ratio)
        val_size = n_time_steps-train_size-test_size
        
        for key in self.user_seq_df.keys():
            if key != 'user_id':
                value = np.stack(self.user_seq_df[key].values)
                self.data_df['train'][key] = value[:, :train_size].tolist()
                self.data_df['val'][key] = value[:, train_size:val_size+train_size].tolist()
                self.data_df['test'][key] = value[:, train_size+val_size:val_size+train_size+test_size].tolist()
                self.data_df['whole'][key] = value[:, :].tolist()

        for key in self.data_df.keys():     
            self.data_df[key] = pd.DataFrame.from_dict(self.data_df[key], orient='columns')   
            self.data_df[key]['user_id'] = self.user_seq_df['user_id']

        
    def show_columns(self):
        """
        Prints a random row of the user sequence DataFrame to show the available data columns.
        """
        # Get a random row from the user sequence DataFrame
        self.logs.write_to_log_file('Data columns:')
        self.logs.write_to_log_file(self.user_seq_df.iloc[np.random.randint(0, len(self.user_seq_df))])



