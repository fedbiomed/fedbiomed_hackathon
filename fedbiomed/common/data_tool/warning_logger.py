from typing import Dict, Any

from fedbiomed.common.data_tool.pre_processing_checks import PreProcessingChecks
from fedbiomed.common.data_tool.pre_processing_warnings_exceptions import  DataSanityCheckException


class WarningReportLogger:
    def __init__(self, disclosure:int):
        self._disclosure = disclosure
        
        
        self._report = {}
        self._current_entry = None
        self._n_warnings = 0
        self._n_sanity_checks = 1
        self._n_exception = 0
        self._n_feature = 1
        self._saved_msg = None
        self._exception_collector = []

        
    def write_new_entry(self, check: PreProcessingChecks):
        self._current_entry = check.name
        if check.name not in self._report:
            
            
            if self._disclosure < 2:
                if isinstance(check, PreProcessingChecks):
                    self._current_entry = 'sanity_check_' + str(self._n_sanity_checks)
                    self._n_sanity_checks += 1
                
                elif issubclass(check.warning_type, Exception):
                    self._current_entry = 'Error_' + str(self._n_exception)
                    self._n_exception += 1
                else:
                    print("input not understood")
            
            self._report[self._current_entry] = []
        
    def write_checking_result(self,
                              success: bool = None,
                              msg: str = '',
                              feature_name: str = '',
                              view_name: str = ''):
        
        _new_entry = {}
        _new_entry['view'] = view_name
        _new_entry['success'] = success
        
        if success :
            msg = "Test passed"
        elif success is None:
            msg = 'Test skipped'
        else:
            self._n_warnings += 1
            print(msg)
        if self._disclosure > 2:
            _new_entry['feature'] = feature_name
            _new_entry['msg'] = msg
        else:
            _new_entry['feature'] = 'feature_' + str(self._n_feature)
            _new_entry['msg'] = ''
            self._n_feature += 1
        
        self._report[self._current_entry].append(_new_entry)

    def get_report(self) -> Dict[str, Any]:
        print(f'number of warnings: {self._n_warnings}\nNumber of error: {self._n_exception}')
        return self._report

    def clean_report(self):
        self._report = {}

    def add_exception(self, exception: Exception):
        self._n_exception += 1
        self._exception_collector.append(exception)
        
    def raise_exception(self):
        if self._exception_collector:
            # case where exception collector is not empty
            raise DataSanityCheckException(self._exception_collector)