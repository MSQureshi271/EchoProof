"""Prospective Option B experiment checks; never official publication."""
import math


def require(ok, message):
    if not ok:
        raise ValueError(message)


def validate_events(events, records=4530, epochs=2, batch=4, accumulation=4):
    attempts_per_epoch = math.ceil(math.ceil(records / batch) / accumulation)
    require(len(events) == epochs * attempts_per_epoch, 'Attempt coverage incomplete')
    previous_scale = 65536.0  # unchanged torch GradScaler default, checked at launch
    summaries = []
    for epoch in range(1, epochs + 1):
        norms = []
        skipped = 0
        for j in range(attempts_per_epoch):
            e = events[(epoch - 1) * attempts_per_epoch + j]
            remain = records - j * batch * accumulation
            size = min(remain, batch * accumulation)
            sizes = [min(batch, size - k) for k in range(0, size, batch)]
            require(e['attempt'] == (epoch - 1) * attempts_per_epoch + j + 1 and e['epoch'] == epoch,
                    'Attempt identity mismatch')
            require(e['microbatch_sizes'] == sizes and e['denominator'] == size, 'Accumulation mathematics changed')
            require('error' not in e and not e['norm_only_overflow'] and not e['scheduler_advanced'],
                    'Unexplained numerical/schedule failure')
            require(e['scale_before'] == previous_scale and math.isfinite(e['scale_after']) and e['scale_after'] > 0,
                    'Scale progression mismatch')
            require(e['lr_before'] == e['lr_after'] == [0.0001], 'Constant head LR changed')
            if e['applied']:
                require(not e['scaler_skipped'] and e['optimizer_step_calls_observed'] == 1
                        and not e['nonfinite_scaled_elements'] and not e['nonfinite_unscaled_elements']
                        and e['preclip_norm'] is not None and math.isfinite(e['preclip_norm']) and e['preclip_norm'] >= 0
                        and e['scale_after'] == previous_scale, 'Unsafe applied update')
                norms.append(e['preclip_norm'])
            else:
                require(e['scaler_skipped'] and e['optimizer_step_calls_observed'] == 0
                        and e['nonfinite_scaled_elements'] and e['nonfinite_unscaled_elements']
                        and e['preclip_norm'] is None and e['scale_after'] == previous_scale * .5,
                        'Unexplained skipped update')
                skipped += 1
            previous_scale = e['scale_after']
        require(norms, 'Epoch has no applied updates')
        summaries.append({'epoch': epoch, 'attempted': attempts_per_epoch, 'applied': len(norms),
                          'scaler_skipped': skipped, 'norm_mean': math.fsum(norms) / len(norms),
                          'norm_min': min(norms), 'norm_max': max(norms)})
    require(any(s['norm_max'] > 0 for s in summaries), 'No finite nonzero applied gradient observed')
    return summaries


def verify_software(report, expected_sources, expected_count):
    require(report.get('software_gate_passed') is True and report.get('returncode') == 0
            and report.get('test_cases') == expected_count
            and report.get('counts') == {'failure': 0, 'error': 0, 'skipped': 0}
            and report.get('verification_inputs_unchanged') is True, 'Software gate missing/failed/skipped')
    require(report.get('expected_source_hashes') == expected_sources, 'Software evidence is stale')
    command = report.get('command', [])
    prefix = ['-B', '-m', 'pytest', 'experiments/r2_u2/tests/test_contract.py',
              'experiments/r2_u2/tests/test_numerical.py', 'experiments/r2_u2/tests/test_launch.py',
              '-q', '-ra', '-p', 'no:cacheprovider', '--basetemp']
    require(isinstance(command, list) and len(command) == 14 and command[1:12] == prefix
            and command[-1].startswith('--junitxml='), 'Software report did not run the complete required suites')
    return True
