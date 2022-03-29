local _base = import '../spider/nl2code-align.jsonnet';

function(args) _base(args + { 
    end_lr: 0,
    setting: 'basic',
    loss_type: 'softmax',

    bs: 20,
    lr: 0.000743552663260837,
    top_k_learnable: 50,
    decoder_recurrent_size: 512,
    decoder_dropout: 0.20687225956012834,
    num_layers: 8,
}) + {
    model_name:: null,

    data: {
        train: {
            name: 'wikisql', 
            paths: [args.data_path + 'train.jsonl'],
            tables_paths: [args.data_path + 'train.tables.jsonl'],
            db_path: args.data_path + 'train.db',
#            limit: 100,
        },
        val: {
            name: 'wikisql', 
            paths: [args.data_path + 'dev.jsonl'],
            tables_paths: [args.data_path + 'dev.tables.jsonl'],
            db_path: args.data_path + 'dev.db',
#            limit: 100,
        },
        test: {
            name: 'wikisql', 
            paths: [args.data_path + 'test.jsonl'],
            tables_paths: [args.data_path + 'test.tables.jsonl'],
            db_path: args.data_path + 'test.db',
#            limit: 100,
        },
    },

    model+: {
        encoder_preproc+: {
            word_emb: {
                name: 'glove',
                kind: '42B',
                lemmatize: true,
            },
            # Only put words needed for top_k_learnable in vocabulary
            max_count: $.model.encoder.top_k_learnable,
            save_path: args.data_path + 'nl2code-wikisql/',
            count_tokens_in_word_emb_for_vocab: true,
            fix_issue_16_primary_keys: true,
        },
        decoder_preproc+: {
            grammar: {
                name: 'wikisql',
            },
            fix_issue_16_primary_keys:: null,
            save_path: args.data_path + 'nl2code-wikisql/',
        },
        decoder+: {
            use_align_loss: false,
        },
    },
}
