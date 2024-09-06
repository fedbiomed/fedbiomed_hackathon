import random
from math import log2
from typing import Union, Optional, List
from fedbiomed.common.exceptions import FedbiomedValueError, FedbiomedTypeError


class Secret:
    def __init__(self, secret: Union[int, List[int]]) -> None:
        """
        Initializes the Secret class with the provided additive secret value.

        Args:
            secret (Union[int, List[int]]): The secret to be shared, either an integer or a list of integers.

        Raises:
            FedbiomedValueError: If the secret is not an int or a list of integers.
        """
        if not (isinstance(secret, int) or (isinstance(secret, list) and all(isinstance(i, int) for i in secret))):
            raise FedbiomedValueError("Secret must be an int or a list of int")
        self.secret = secret

    def split(self, num_shares: int, bit_length: Optional[int] = None) -> 'Shares':
        """
        Splits the secret into the specified number of shares using additive secret sharing.
        The sum of the shares will equal the original secret.

        Args:
            num_shares (int): The number of shares to generate.
            bit_length (Optional[int], optional): The bit length of the shares. Defaults to None.

        Returns:
            Shares: A Shares object representing the split shares.

        Raises:
            FedbiomedValueError: If the number of shares is less than or equal to 0.
            FedbiomedValueError: If the bit length is smaller than the secret's bit length.
        """
        if num_shares <= 0:
            raise FedbiomedValueError("Number of shares must be greater than 0")
        if isinstance(self.secret, int):
            if bit_length is not None and bit_length < int(log2(self.secret)):
                raise FedbiomedValueError("Bit length must be greater or equal than the secret's bit length")
            bit_length = self.secret.bit_length() if bit_length is None else bit_length
            shares = [random.randint(0, 2 ** bit_length) for _ in range(num_shares - 1)]
            last_share = self.secret - sum(shares)
            shares.append(last_share)
        else:
            shares = []
            max_value = max(self.secret)
            if bit_length is not None and bit_length < int(log2(max_value)):
                raise FedbiomedValueError("Bit length must be greater or equal than the largest secret's bit length")
            bit_length = max_value.bit_length() if bit_length is None else bit_length
            for value in self.secret:
                partial_shares = [random.randint(0, 2 ** bit_length) for _ in range(num_shares - 1)]
                partial_shares.append(value - sum(partial_shares))
                shares.append(partial_shares)
            shares = list(map(list, zip(*shares)))

        return Shares([Share(share) for share in shares])


class Shares:
    """A class to represent a collection of Share objects."""

    def __init__(self, shares: List['Share']) -> None:
        """
        Initializes the Shares class with a list of Share objects.

        Args:
            shares (List[Share]): A list of Share objects.

        Raises:
            FedbiomedTypeError: If the shares are not of type Share.
        """
        if not all(isinstance(share, Share) for share in shares):
            raise FedbiomedTypeError("All shares must be of type Share")
        self.shares = shares

    def __add__(self, other: 'Shares') -> 'Shares':
        """
        Adds two Shares objects together.

        Args:
            other (Shares): The Shares object to add.

        Returns:
            Shares: A new Shares object with the resulting sums.

        Raises:
            FedbiomedTypeError: If the two Shares objects are not of the same length.
        """
        if len(self.shares) != len(other.shares):
            raise FedbiomedTypeError("Shares must be of the same length")
        # Check if both lists contain shares of the same type
        if all(isinstance(share.value, int) for share in self.shares) != all(isinstance(share.value, int) for share in other.shares):
            raise FedbiomedTypeError("Shares must be of the same type")
        elif all(isinstance(share.value, int) for share in self.shares):
            result = Shares([self.shares[i] + other.shares[i] for i in range(len(self.shares))])
        elif all(isinstance(share.value, list) for share in self.shares):
            result = Shares([Share([self.shares[i].value[j] + other.shares[i].value[j] for j in range(len(self.shares[i].value))]) for i in range(len(self.shares))])
        
        else:
            raise FedbiomedTypeError("Shares must be of the same type")
            
        return result
    
    @property
    def values(self) -> List[Union[int, List[int]]]:
        """
        Getter for the values of the shares.

        Returns:
            List[Union[int, List[int]]]: The values of the shares.
        """
        return [share.value for share in self.shares]
    
    def reconstruct(self) -> Union[int, List[int]]:
        """
        Reconstructs the secret from the shares.

        Returns:
            Union[int, List[int]]: The reconstructed secret.
        """
        if all(isinstance(share.value, int) for share in self.shares):
            result = sum(share.value for share in self.shares)
        elif all(isinstance(share.value, list) for share in self.shares):
            result = [sum(share.value[i] for share in self.shares) for i in range(len(self.shares[0].value))]
        else:
            raise FedbiomedTypeError("Shares must be of the same type")
        return result

    @property
    def values(self) -> List[Union[int, List[int]]]:
        """
        Getter for the values of the shares.

        Returns:
            List[Union[int, List[int]]]: The values of the shares.
        """
        return [share.value for share in self.shares]
    
    def __getitem__(self, index: int) -> 'Share':
        """
        Retrieves a Share object from the Shares object.

        Args:
            index (int): The index of the Share object to retrieve.

        Returns:
            Share: The Share object at the specified index.
        """
        return self.shares[index]

class Share:
    def __init__(self, value: Union[int, List[int]]) -> None:
        """
        Initializes the Share class with a given value, representing an additive secret share.

        Args:
            value (Union[int, List[int]]): The value of the share, either an integer or a list of integers.

        Raises:
            FedbiomedTypeError: If the value is neither an int nor a list of integers.
        """
        if not (isinstance(value, int) or (isinstance(value, list) and all(isinstance(i, int) for i in value))):
            raise FedbiomedTypeError("Share value must be an int or a list of int")
        self._value = value

    def __add__(self, other: 'Share') -> 'Share':
        """
        Adds two shares together. Supports both integer and list types.

        Args:
            other (['Share']): The share or value to add.

        Returns:
            Share: A new Share object with the resulting sum.

        Raises:
            FedbiomedTypeError: If the two values being added are not of the same type (both int or both list).
        """
        if isinstance(other, Share):
            if isinstance(self.value, int) and isinstance(other.value, int):
                result = self.value + other.value
            elif isinstance(self.value, list) and isinstance(other.value, list):
                result = [self.value[i] + other.value[i] for i in range(len(self.value))]
            else:
                raise FedbiomedTypeError("Shares must be of the same type")
        return Share(result)

    def __repr__(self) -> str:
        return f"Share({self.value})"

    @property
    def value(self) -> Union[int, List[int]]:
        """
        Getter for the share's value.

        Returns:
            Union[int, List[int]]: The value of the share.
        """
        return self._value