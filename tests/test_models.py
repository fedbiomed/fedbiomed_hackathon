import unittest
import urllib.request
import logging
from unittest.mock import MagicMock, patch, mock_open
from fedbiomed.common.logger import logger
from fedbiomed.common.models.model import BaseSkLearnModel, SGDClassiferSKLearnModel, SGDRegressorSKLearnModel, SkLearnModel, capture_stdout
from sklearn.linear_model import SGDClassifier, SGDRegressor
import numpy as np


class TestDocumentationLinks(unittest.TestCase):
    skip_internet_test: bool
    def setUp(self) -> None:
        # test internet connection by reaching google website
        google_url = 'http://www.google.com'
        try:
            url_res = urllib.request.urlopen(google_url)
        except urllib.error.URLError as err:
            self.skip_internet_test = True
            return
        if url_res.code != 200:
            self.skip_internet_test = True
        else:
            self.skip_internet_test = False
    
    def tearDown(self) -> None:
        pass
    
    def test_testdocumentationlinks_01(self):
        links = (
            'https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html',
        )
        if self.skip_internet_test:
            self.skipTest("no internet connection: skipping test_testdocumentationlinks_01")
        for link in links:
            url_res = urllib.request.urlopen(link)
            self.assertEqual(url_res.code, 200, f"cannot reach url link {link} pointed in documentation")


class TestSkLearnModel(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)
        self.sgdclass_model = SkLearnModel(SGDClassifier)
        self.sgdregressor_model = SkLearnModel(SGDRegressor)
        
    def tearDown(self) -> None:
        logging.disable(logging.NOTSET)
    
    def test_sklearnmodel_method_01_save(self):
        saved_params = []
        def mocked_joblib_dump(obj, *args, **kwargs):
            saved_params.append(obj)
        
        self.sgdclass_model.set_weights({'coef_': 0.42, 'intercept_': 0.42})
        with patch('fedbiomed.common.models.model.joblib.dump',
                   side_effect=mocked_joblib_dump), \
                patch('builtins.open', mock_open()):
            self.sgdclass_model.save('filename')
            self.assertEqual(saved_params[-1].coef_, 0.42)
            self.assertEqual(saved_params[-1].intercept_, 0.42)
            
    def test_sklearnmodel_method_02_load(self):
        self.sgdclass_model.set_init_params({'n_classes':3, 'n_features':5})
        with (
            # patch.object(self.sgdclass_model, 'param_list', ['coef_', 'intercept_']), 
                patch.object(self.sgdclass_model.model, 'coef_', 0.42), 
                patch.object(self.sgdclass_model.model, 'intercept_',  0.42), 
                patch('fedbiomed.common.models.model.joblib.load',
                    return_value=self.sgdclass_model.model), 
                patch('builtins.open', mock_open())
                ):
            self.sgdclass_model.load('filename')
            self.assertDictEqual(self.sgdclass_model.get_weights(), {'coef_': 0.42, 'intercept_': 0.42})

    def test_sklearnmodel_03_set_init_params(self):
        # self.assertEqual(training_plan._model.n_iter_, 1)
        # test several values for `model_args`
        model_args_iterator = (
            {'n_classes': 2, 'n_features': 1},
            {'n_classes': 2, 'n_features': 2},
            {'n_classes': 3, 'n_features': 1},
            {'n_classes': 3, 'n_features': 3}
        )
        
        for model_args in model_args_iterator:
            self.sgdclass_model.set_init_params(model_args)
            self.assertListEqual(sorted(self.sgdclass_model.param_list), sorted(['coef_', 'intercept_']))
        
class TestSklearnTrainingPlansClassification(unittest.TestCase):
    implemented_models = [SGDClassifier]  # store here implemented model
    model_args = {
        SGDClassifier: {'max_iter': 4242, 'alpha': 0.999, 'n_classes': 2, 'n_features': 2, 'key_not_in_model': None},
    }
    expected_params_list = {
        SGDClassifier: ['intercept_', 'coef_'],

    }

    def setUp(self):     

        logging.disable('CRITICAL')  # prevent flood of messages about missing datasets

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_model_sklearnclassification_01_parameters(self):
        # TODO: add testing additional parameters (set_learning_rate/get_learning_rate)
        
        for model in self.implemented_models:
            # dual classes
            sk_model = SkLearnModel(model)
            self.assertTrue(sk_model._is_classification)
            sk_model.set_init_params(model_args={'n_classes':2, 'n_features': 5})
            # Parameters all initialized to 0.
            for key in sk_model.param_list:
                self.assertTrue(np.all(getattr(sk_model.model, key) == 0.))
                self.assertEqual(getattr(sk_model.model, key).shape[0], 1)
            # Test that classes values are integers in the range [0, n_classes)
            for i in np.arange(2):
                self.assertEqual(sk_model.model.classes_[i], i)

            # Multiclass (class=3):

            multiclass_model_args = {
                **TestSklearnTrainingPlansClassification.model_args[model],
                'n_classes': 3
            }
            sk_model = SkLearnModel(model)
            sk_model.set_init_params(multiclass_model_args)
            # Parameters all initialized to 0.
            for key in sk_model.param_list:
                self.assertTrue(np.all(getattr(sk_model.model, key) == 0.),
                                f"{model.__class__.__name__} Multiclass did not initialize all parms to 0.")
                self.assertEqual(getattr(sk_model.model, key).shape[0], 3,
                                 f"{model.__class__.__name__} Multiclass wrong shape for {key}")
            # Test that classes values are integers in the range [0, n_classes)
            for i in np.arange(3):
                self.assertEqual(sk_model.model.classes_[i], i,
                                 f"{model.__class__.__name__} Multiclass wrong values for classes")


    def test_model_sklearnclassification_03_losses(self):
        def fake_context_manager(value):
            # mimics context_manager
            class MockContextManager:
                return_value = None
                def __init__(self, *args, **kwargs):
                    pass
                def __enter__(self):
                    return value
                def __exit__(self, type, value, traceback):
                    pass
            return MockContextManager()
        
        for model in self.implemented_models:
            
            sk_model = SkLearnModel(model)
            sk_model.model_args = {'n_classes': 3}
            sk_model.model.classes_ = np.array([0, 1, 2])
            sk_model.set_init_params({'n_classes': 3, 'n_features': 2})

            sk_model.init_training()
            actual_losses_stdout = [
                    ['loss: 1.0', 'loss: 0.0', 'loss: 2.0'],
                    ['loss: 0.0', 'loss: 1.0', 'epoch', 'loss: 0.0'],
            ]
            
            inputs = np.array([[1, 2], [1, 1],[0, 1]])
            target = np.array([0, 2, 1])
            
            
            context_manager_patcher = patch('fedbiomed.common.models.model.capture_stdout',
                                            return_value=fake_context_manager(actual_losses_stdout))
            
            collected_losses_stdout = []
            context_manager_patcher.start()
            sk_model.train(inputs, target, collected_losses_stdout)
            context_manager_patcher.stop()

            
            self.assertListEqual(collected_losses_stdout, [actual_losses_stdout])


if __name__ == '__main__':  # pragma: no cover
    unittest.main()

# Test init params