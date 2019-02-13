import os

class Config:
    ''' Neural net hyperparameters '''

    gpu=True
    model_jointness = False
    mention_rep = True


    ''' data hyperparameters '''
    def __init__(self, run_dir, args):

        self.struct_weight = args.struct_weight
        self.dropout = args.dropout
        self.dataset = args.dataset
        self.encoder = args.encoder
        self.args = args
        self.struct_weight = args.struct_weight
        self.typing_weight = args.typing_weight
        self.lr = args.lr
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
        self.parent_sample_size = args.parent_sample_size
        self.bilinear_l2 = args.bilinear_l2


        self.distance_func = args.distance_func

        self.features = args.features
        self.complex  = args.complex
        self.asymmetric = args.asymmetric
        self.base_dir = args.base_dir
        base_dir = args.base_dir

        self.embedding_file = "%s/data/pretrained_embeddings.npz" %base_dir
        self.embedding_downloaded_file = "%s/other_resources/glove.840B.300d.txt" %base_dir
        self.crosswikis_file = "/iesl/canvas/nmonath/data/crosswikis/dictionary.bz2"
        self.redirects_file = "/iesl/canvas/smurty/wiki-data/enwiki-20160920-redirect.tsv"

        if self.dataset == "figer":
            self.train_file="%s/wiki_typing/wiki_train_types"%base_dir
            self.dev_file="%s/wiki_typing/dev"%base_dir
            self.test_file="%s/wiki_typing/test"%base_dir

            self.entity_file="%s/AIDA_linking/entities.joblib"%base_dir
            self.type_file="%s/wiki_typing/figer_type2Idx.joblib"%base_dir
            self.entity_type_file="%s/wiki_typing/entity_type_map.joblib"%base_dir
            self.typenet_matrix = "%s/types_annotated/figer_hierarchy.joblib"%base_dir

            self.raw_entity_file="%s/AIDA_linking/AIDA_original/all_entities.txt"%base_dir
            self.raw_type_file="%s/AIDA_linking/figer_types.txt"%base_dir
            self.raw_entity_type_file="%s/AIDA_linking/AIDA_original/all_entities_types_figer.txt"%base_dir
            self.cross_wikis_shelve = "%s/AIDA_linking/crosswikis.shelve" %base_dir

            self.feature_file = "%s/wiki_typing/feature2id_figer.txt" %feature_file

            self.num_epochs = 20
            self.batch_size = 2048

        elif self.dataset == "typenet":
            self.train_file="%s/AIDA_linking/wiki_train"%base_dir
            self.dev_file="%s/AIDA_linking/dev_processed"%base_dir
            self.test_file="%s/AIDA_linking/test_processed"%base_dir

            self.entity_file="%s/AIDA_linking/entities.joblib"%base_dir
            self.type_file="%s/AIDA_linking/types.joblib"%base_dir
            self.entity_type_file="%s/AIDA_linking/entity_type_map.joblib"%base_dir
            self.typenet_matrix = "%s/types_annotated/typenet_matrix.joblib"%base_dir

            self.raw_entity_file="%s/AIDA_linking/AIDA_original/all_entities.txt"%base_dir
            self.raw_type_file="%s/types_annotated/typenet_structure.txt"%base_dir
            self.raw_entity_type_file="%s/AIDA_linking/AIDA_original/all_entities_types.txt"%base_dir
            self.cross_wikis_shelve = "%s/AIDA_linking/crosswikis.shelve" %base_dir

        elif self.dataset == "umls":
            self.train_file = "%s/meta_data_processed/meta_train_transitive.joblib"%base_dir
            self.dev_file   = "%s/meta_data_processed/meta_train.joblib"%base_dir
            self.test_file  = "%s/meta_data_processed/meta_train.joblib"%base_dir

            self.entity_file = "%s/meta_data_processed/entities.joblib"%base_dir
            self.type_file   = "%s/meta_data_processed/types.joblib"%base_dir
            self.cross_wikis_shelve = "%s/meta_data_processed/crosswikis.shelve" %base_dir
            self.hierarchy = "%s/meta_data_processed/entity_hierarchy.joblib" %base_dir
            self.hierarchy_orig = "%s/meta_data_processed/entity_hierarchy_orig.joblib" %base_dir


        self.vocab_file="%s/data/vocab.joblib"%base_dir
        self.checkpoint_file = "%s/checkpoints_figer_final" %run_dir
        self.model_file = args.model_name


