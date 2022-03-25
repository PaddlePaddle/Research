local _base = import 'nl2code-base.libsonnet';
local _output_from = true;

function(args) _base(output_from=_output_from, data_path=args.data_path) + {
    local lr_s = '%0.1e' % args.lr,
    local end_lr_s = if args.end_lr == 0 then '0e0' else '%0.1e' % args.end_lr,

    model_name: 'bs=%(bs)d,lr=%(lr)s,end_lr=%(end_lr)s,att=%(att)d' % (args + {
        lr: lr_s,
        end_lr: end_lr_s,
    }),

    model+: {
        encoder+: {
            batch_encs_update: false,
            top_k_learnable: args.top_k_learnable,
            update_config+:  {
                num_layers: args.num_layers,
                sc_link: true,
            },
        },
        encoder_preproc+: {
            word_emb+: {
                lemmatize: true,
            },
            min_freq: 4,
            max_count: 5000,
            save_path: args.data_path + 'nl2code-align,output_from=%s,emb=glove-42B,min_freq=%s/' % [_output_from, self.min_freq],
        },
        decoder_preproc+: {
            grammar+: {
                end_with_from: true,
            },
            save_path: args.data_path + 'nl2code-align,output_from=%s,emb=glove-42B,min_freq=%s/' % [_output_from, self.min_freq],
        },
        decoder+: {
            loss_type: args.loss_type,
            recurrent_size : args.decoder_recurrent_size,
            dropout : args.decoder_dropout,
            use_align_mat: true,
            use_align_loss: true,
        },
        log: {
            reopen_to_flush:true,
        },
    },

    train+: {
        batch_size: args.bs,

        model_seed: args.att,
        data_seed:  args.att,
        init_seed:  args.att,
    },

    lr_scheduler+: {
        start_lr: args.lr,
        end_lr: args.end_lr,
    },

}

