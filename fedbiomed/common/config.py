import configparser
import os 
import uuid 

from abc import ABC

from typing import Optional

from fedbiomed.common.constants import ErrorNumbers, MPSPDZ_certificate_prefix, CONFIG_FOLDER_NAME
from fedbiomed.common.utils import  raise_for_version_compatibility, CONFIG_DIR, ROOT_DIR
from fedbiomed.common.certificate_manager import retrieve_ip_and_port, generate_certificate
from fedbiomed.common.constants import SERVER_certificate_prefix, \
    __researcher_config_version__, __node_config_version__, \
    HashingAlgorithms, \
    ETC_FOLDER_NAME, \
    VAR_FOLDER_NAME, \
    DB_PREFIX



class Config(ABC):
    """Base Config class"""

    def __init__(
        self,
        root = None,
        name: Optional[str] = None,
        auto_generate: bool = True
    ) -> None:
        """Initializes config"""
        self.root = root
        self._cfg = configparser.ConfigParser()
        self.name = name if name \
            else os.getenv("CONFIG_FILE", self.DEFAULT_CONFIG_FILE_NAME)

        if self.root:
            self.path = os.path.join(self.root, CONFIG_FOLDER_NAME, self.name)
            self.root = self.root
        else:
            self.path = os.path.join(CONFIG_DIR, self.name)
            self.root = ROOT_DIR

        if auto_generate:
            self.generate()

    def is_config_existing(self):
        """Checks if config file is exsiting"""

        return os.path.isfile(self.path)

    def read(self):
        """Reads configuration file that is already existing in given path"""

        self._cfg.read(self.path)

        # Validate config version
        raise_for_version_compatibility(
            self._cfg["default"]["version"],
            self.CONFIG_VERSION,
            f"Configuration file {self.path}: found version %s expected version %s")

        return True

    def get(self, section, key) -> str:
        """Returns value for given ket and section"""

        return self._cfg.get(section, key)

    def sections(self) -> list:
        """Returns sections of the config"""

        return self._cfg.sections()

    def generate(self, force: bool = False) -> bool:
        """"Generate configuration file

        Args:
        force: Overwrites existing configration file
        """

        if self.is_config_existing() and not force:
            return self.read()

        component_id = self._cfg['default']['id']
        self._cfg['default']['component'] = self.COMPONENT_TYPE 
        self._cfg['default']['version'] = str(self.CONFIG_VERSION)
        # DB PATH RELATIVE
        db_path  = os.path.join(self.root, VAR_FOLDER_NAME, f"{DB_PREFIX}{component_id}.json")
        self._cfg['default']['db'] = os.path.relpath(db_path, os.path.join(self.root, ETC_FOLDER_NAME))

        ip, port = retrieve_ip_and_port(self.root)
        allow_default_biprimes = os.getenv('ALLOW_DEFAULT_BIPRIMES', True)

        # Generate self-signed certificates
        key_file, pem_file = generate_certificate(
            root=self.root, 
            component_id=component_id, 
            prefix=MPSPDZ_certificate_prefix)

        self._cfg['mpspdz'] = {
            'private_key': os.path.relpath(key_file, os.path.join(self.root, 'etc')),
            'public_key': os.path.relpath(pem_file, os.path.join(self.root, 'etc')),
            'mpspdz_ip': ip,
            'mpspdz_port': port,
            'allow_default_biprimes': allow_default_biprimes,
            'default_biprimes_dir': os.path.relpath(
                os.path.join(self.root, 'envs', 'common', 'default_biprimes'),
                os.path.join(self.root, 'etc')
            )
        }

        try:
            with open(self.path, 'w') as f:
                self._cfg.write(f)
        except configparser.Error:  
            raise IOError(ErrorNumbers.FB600.value + ": cannot save config file: " + self.path)

        return True


class NodeConfig(Config):

    DEFAULT_CONFIG_FILE_NAME: str = 'config_node.ini'
    COMPONENT_TYPE: str = 'NODE'
    CONFIG_VERSION: str = __node_config_version__

    def generate(self, force: bool = False):
        """Generate researcher config"""

        node_id = os.getenv('NODE_ID', 'node_' + str(uuid.uuid4()))
        self._cfg['default'] = {'id': node_id}


        # Security variables
        self._cfg['security'] = {
            'hashing_algorithm': HashingAlgorithms.SHA256.value,
            'allow_default_training_plans': os.getenv('ALLOW_DEFAULT_TRAINING_PLANS', True),
            'training_plan_approval': os.getenv('ENABLE_TRAINING_PLAN_APPROVAL', False),
            'secure_aggregation': os.getenv('SECURE_AGGREGATION', True),
            'force_secure_aggregation': os.getenv('FORCE_SECURE_AGGREGATION', False)
        }

        # gRPC server host and port
        self._cfg["researcher"] = {
            'ip': os.getenv('RESEARCHER_SERVER_HOST', 'localhost'),
            'port': os.getenv('RESEARCHER_SERVER_PORT', '50051')
        }

        return super().generate(force)


class ResearcherConfig(Config):

    DEFAULT_CONFIG_FILE_NAME: str = 'config_researcher.ini'
    COMPONENT_TYPE: str = 'RESEARCHER'
    CONFIG_VERSION: str = __researcher_config_version__

    def generate(self, force: bool = False):
        """Generate researcher config"""

        researcher_id = os.getenv('RESEARCHER_ID', 'researcher_' + str(uuid.uuid4()))
        self._cfg['default'] = { 'id': researcher_id }

        grpc_host = os.getenv('RESEARCHER_SERVER_HOST', 'localhost')
        grpc_port = os.getenv('RESEARCHER_SERVER_PORT', '50051')

        # Generate certificate for gRPC server
        key_file, pem_file = generate_certificate(
            root=self.root, 
            component_id=researcher_id, 
            prefix=SERVER_certificate_prefix,
            subject={'CommonName': grpc_host}
        )

        self._cfg['server'] = {
            'host': grpc_host,
            'port': grpc_port,
            'pem' : os.path.relpath(pem_file, os.path.join(self.root, ETC_FOLDER_NAME)),
            'key' : os.path.relpath(key_file, os.path.join(self.root, ETC_FOLDER_NAME))
        }

        return super().generate(force)
