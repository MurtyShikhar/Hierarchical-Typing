import os

class Config_MIL:
    ''' Neural net hyperparameters '''

    gpu=True
    mention_rep = True


    ''' data hyperparameters '''



    def __init__(self, run_dir, args):

        self.dropout = args.dropout
        self.dataset = args.dataset
        self.args = args

        self.bag_size = args.bag_size

        self.lr = args.lr
        self.encoder = args.encoder
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.epsilon = args.epsilon
        self.embedding_dim = args.embedding_dim
        self.hidden_dim = args.hidden_dim
        self.kernel_width = args.kernel_width
        self.clip_val = args.clip_val
        self.weight_decay = args.weight_decay
        self.num_epochs = args.num_epochs
        self.batch_size = args.batch_size
        self.save_model = args.save_model
        self.struct_weight = args.struct_weight
        self.linker_weight = args.linker_weight
        self.typing_weight = args.typing_weight


        self.complex = args.complex
        self.mode = args.mode
        self.bilinear_l2 = args.bilinear_l2
        self.parent_sample_size = args.parent_sample_size
        self.asymmetric = args.asymmetric

        self.test_batch_size = args.test_batch_size
        self.take_frac = args.take_frac
        self.use_transitive = args.use_transitive
        self.base_dir = args.base_dir
        base_dir = args.base_dir

        # no support for struct weight with figer
        if self.dataset == "figer":
            self.test_bag_size = 1
            self.type_dict        = "%s//MIL_data/figer_type_dict.joblib" %base_dir
            self.typenet_matrix   = "%s/types_annotated/TypeNet_transitive_closure.joblib" %base_dir
            self.dev_file="%s/wiki_typing/dev"%base_dir
            self.test_file="%s/wiki_typing/test"%base_dir

        else:
            self.test_bag_size = 20
            

            self.type_dict        = "%s/types_annotated/TypeNet_type2idx.joblib" % base_dir
            self.typenet_matrix   = "%s/types_annotated/TypeNet_transitive_closure.joblib" % base_dir

            self.train_file = "%s/MIL_data/train.entities"%base_dir
            self.dev_file   = "%s/MIL_data/dev.entities" % base_dir
            self.test_file  = "%s/MIL_data/test.entities" % base_dir
            self.bag_file   = "%s/MIL_data/entity_bags.joblib" % base_dir
            self.entity_dict = "%s/MIL_data/entity_dict.joblib" % base_dir
            self.cross_wikis_shelve = "%s/MIL_data/alias_table.joblib" %base_dir



        self.embedding_file = "%s/data/pretrained_embeddings.npz" %base_dir
        self.embedding_downloaded_file = "%s/other_resources/glove.840B.300d.txt" %base_dir
        self.crosswikis_file = "%s/other_resources/dictionary.bz2"
        self.redirects_file = "/iesl/canvas/smurty/wiki-data/enwiki-20160920-redirect.tsv"


        self.entity_bags_dict = "%s/MIL_data/entity_bags.joblib" %base_dir

        if self.use_transitive:
            self.entity_type_dict = "%s/MIL_data/entity_%s_type_dict.joblib" %(base_dir, self.dataset)
        else:
            self.entity_type_dict = "%s/MIL_data/entity_type_dict_orig.joblib" %(base_dir)

        self.entity_type_dict_test = "%s/MIL_data/entity_%s_type_dict.joblib" %(base_dir, self.dataset)


        self.raw_entity_file="%s/AIDA_linking/AIDA_original/all_entities.txt"%base_dir
        self.raw_type_file="%s/types_annotated/typenet_structure.txt"%base_dir
        self.raw_entity_type_file="%s/AIDA_linking/AIDA_original/all_entities_types.txt"%base_dir


        self.vocab_file="%s/data/vocab.joblib"%base_dir
        self.checkpoint_file = "%s/checkpoints" %run_dir
        self.model_file = args.model_name
