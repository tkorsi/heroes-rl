"""Cohort reporting checks only; no engine process or simulated game adapter."""
import unittest

from validation.probes.validate_contextual_random import cohort_exit_status, cohort_report


def row(status, reason, *, audited=True, error=None):
    return {'status': status, 'termination_reason': reason, 'audit_passed': audited, 'error': error}


class ContextualRandomReportingTests(unittest.TestCase):
    def test_verified_natural_and_cutoff_cases_succeed_without_inventing_outcomes(self):
        report = cohort_report([row('completed', 'natural_terminal'), row('truncated', 'collection_window')])
        self.assertEqual(cohort_exit_status(report), 0)
        self.assertEqual(report['natural_completions'], 1)
        self.assertEqual(report['supported_cutoffs'], 1)
        self.assertEqual(report['failed_cases'], 0)
        self.assertEqual(report['native_status_counts'],
                         {'completed': 1, 'truncated': 1, 'failed': 0, 'unfinalized': 0})

    def test_native_failure_fails_cohort_even_when_failure_history_audits(self):
        report = cohort_report([row('failed', 'unsupported_interface')])
        self.assertEqual(cohort_exit_status(report), 1)
        self.assertEqual(report['native_failures'], 1)
        self.assertEqual(report['probe_exceptions'], 0)
        self.assertEqual(report['failed_cases'], 1)
        self.assertEqual(report['termination_reason_counts'], {'unsupported_interface': 1})
        self.assertEqual(report['natural_completions'], 0)
        self.assertEqual(report['supported_cutoffs'], 0)

    def test_probe_exception_cannot_pass_through_successful_cleanup_cutoff(self):
        report = cohort_report([row('truncated', 'collection_window', error='ValueError: changed delivered prefix')])
        self.assertEqual(cohort_exit_status(report), 1)
        self.assertEqual(report['native_failures'], 0)
        self.assertEqual(report['probe_exceptions'], 1)
        self.assertEqual(report['failed_cases'], 1)
        self.assertEqual(report['supported_cutoffs'], 0)

    def test_unverified_or_unfinalized_case_is_not_successful(self):
        for case in (row('completed', 'natural_terminal', audited=False), row(None, None, audited=False)):
            with self.subTest(case=case):
                report = cohort_report([case])
                self.assertEqual(cohort_exit_status(report), 1)
                self.assertEqual(report['natural_completions'], 0)
                self.assertEqual(report['failed_cases'], 1)

    def test_native_failure_and_probe_exception_are_counted_once_as_failed_case(self):
        report = cohort_report([row('failed', 'engine_error', error='ValueError: audit failed'),
                                row('failed', 'engine_error')])
        self.assertEqual(report['native_failures'], 2)
        self.assertEqual(report['probe_exceptions'], 1)
        self.assertEqual(report['failed_cases'], 2)
        self.assertEqual(report['termination_reason_counts'], {'engine_error': 2})


if __name__ == '__main__':
    unittest.main()
