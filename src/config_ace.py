class Config:
    ''' Neural net hyperparameters '''
    num_epochs = 10
    batch_size = 256
    hidden_state_size=150
    embedding_dim=300
    clip_val=5


    ''' data hyperparameters '''
    base_dir = "/iesl/canvas/smurty/epiKB"

    embedding_file = "%s/data/pretrained_embeddings.npz" %base_dir
    embedding_downloaded_file = "%s/other_resources/glove.840B.300d.txt" %base_dir 
    vocab_file="%s/ACE_linking/vocab.joblib"%base_dir    
    entity_file="%s/ACE_linking/entities.joblib"%base_dir
    raw_entity_file = "%s/ACE_linking/ACE_entity_file.txt"%base_dir
    raw_vocab_file = "/iesl/canvas/smurty/wiki-data/all_vocab/vocab_fast.tsv"
    train_file="%s/ACE_linking/train_processed"%base_dir
    dev_file="%s/ACE_linking/dev_processed"%base_dir
    test_file="%s/ACE_linking/test_processed"%base_dir
    cross_wikis_shelve = "/iesl/canvas/smurty/epiKB/data/crosswikis/crosswikis.shelve"
    crosswikis_file = "/iesl/canvas/nmonath/data/crosswikis/dictionary.bz2"

    checkpoint_file = "%s/checkpoints" %base_dir
    model_file = "lstm_entity_linker_ace"