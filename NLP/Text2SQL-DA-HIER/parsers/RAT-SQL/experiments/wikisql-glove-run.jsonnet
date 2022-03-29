{
    logdir: "logdir/glove_run",
    model_config: "configs/wikisql/nl2code-wikisql.jsonnet",
    model_config_args: {
        att: 0,
        data_path: 'data/wikisql/',
    },

    eval_name: "wikisql_glove_run_%s_%d" % [self.eval_use_heuristic, self.eval_beam_size],
    eval_output: "__LOGDIR__/ie_dirs",
    eval_beam_size: 1,
    eval_use_heuristic: true,
    eval_steps: [ 1000 * x + 100 for x in std.range(30, 39)] + [40000],
    eval_section: "val",
}
