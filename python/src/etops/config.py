"""
Configuration for tensor operations.

This module defines TensorOperationConfig, the main configuration dataclass
for specifying tensor operations across different backends.
"""

from dataclasses import dataclass
from typing import Sequence, Optional
import json

from etops.types import DataType, PrimType, ExecType, DimType


@dataclass(frozen=True)
class TensorOperationConfig:
    """
    Configuration for tensor operations.

    Supports both binary contractions and unary operations.
    The operation type is automatically determined from prim_main.

    The strides parameter is a 3D tensor with shape [LEVEL][TENSOR][DIMENSION]:

    LEVEL dimension (axis 0):
      - [0] = Primary memory layout strides
      - [1] = Secondary level (e.g., packing strides, L1 cache)
      - [2+] = Additional levels (e.g., L2, L3 cache)

      For user input, typically only level 0 is provided.
      The optimizer may add additional levels (e.g., packing).

    TENSOR dimension (axis 1):
      - Binary operations: [0]=in0, [1]=in1, [2]=out
      - Unary operations: [0]=in, [1]=out

    DIMENSION dimension (axis 2):
      - Corresponds to dimension indices

    Binary Contractions:
      - backend: "tpp" or "tileir"
      - prim_main: etops.prim.gemm or etops.prim.brgemm
      - dim_types: combination of etops.dim.m, .n, .k, .c
      - prim_first: etops.prim.zero or .none (optional first touch)
      - prim_last: etops.prim.relu or .none (optional last touch)
      - strides: shape [1 or more][3][num_dims]

    Unary Operations:
      - backend: "tpp" or "tileir"
      - prim_main: etops.prim.copy or .relu
      - dim_types: must be etops.dim.c for all dimensions
      - prim_first: must be etops.prim.none
      - prim_last: must be etops.prim.none
      - strides: shape [1][2][num_dims]
    """

    backend: str
    data_type: DataType
    prim_first: PrimType
    prim_main: PrimType
    prim_last: PrimType
    dim_types: Sequence[DimType]
    exec_types: Sequence[ExecType]
    dim_sizes: Sequence[int]
    strides: Sequence[Sequence[Sequence[int]]]  # [LEVEL][TENSOR][DIMENSION]

    def __post_init__(self):
        """Validate configuration at creation time."""
        # Validate backend is explicitly set
        valid_backends = {"tpp", "tileir"}
        if self.backend not in valid_backends:
            raise ValueError(
                f"Unsupported backend: '{self.backend}'. "
                f"Supported backends: {sorted(valid_backends)}."
            )

        # Validate data type compatibility with backend
        tpp_supported_dtypes = {DataType.float32, DataType.float64}
        if self.backend == "tpp" and self.data_type not in tpp_supported_dtypes:
            raise ValueError(
                f"TPP backend only supports float32 and float64 data types. "
                f"Got: {self.data_type.name}. "
                f"Use tileir backend for float16, bfloat16, and tfloat32."
            )

        # Determine operation type from prim_main
        is_binary = self.prim_main in [PrimType.gemm, PrimType.brgemm]
        is_unary = self.prim_main in [PrimType.copy, PrimType.relu]

        if self.prim_main == PrimType.zero:
            raise ValueError(
                "prim_main=prim.zero is not a valid unary operation. "
                "Use prim_first=prim.zero for beta=0 accumulator init in "
                "binary contractions, or prim_main=prim.copy/prim.relu "
                "for unary operations."
            )

        if not is_binary and not is_unary:
            raise ValueError(
                f"Invalid prim_main: {self.prim_main}. "
                f"Must be gemm/brgemm (binary) or copy/relu (unary)."
            )

        # Validate dimension consistency
        num_dims = len(self.dim_types)
        if len(self.exec_types) != num_dims:
            raise ValueError(
                f"exec_types length ({len(self.exec_types)}) must match "
                f"dim_types length ({num_dims})."
            )
        if len(self.dim_sizes) != num_dims:
            raise ValueError(
                f"dim_sizes length ({len(self.dim_sizes)}) must match "
                f"dim_types length ({num_dims})."
            )

        # Validate dim_sizes are positive
        for i, size in enumerate(self.dim_sizes):
            if size <= 0:
                raise ValueError(f"dim_sizes[{i}] must be positive, got {size}.")

        # Validate strides shape
        if len(self.strides) == 0:
            raise ValueError("strides must have at least one level (level 0).")

        # Validate level 0 strides
        expected_tensors = 3 if is_binary else 2
        if len(self.strides[0]) != expected_tensors:
            raise ValueError(
                f"strides[0] must have {expected_tensors} tensors "
                f"({'in0, in1, out' if is_binary else 'in, out'}), "
                f"got {len(self.strides[0])}."
            )

        # Validate stride dimensions match num_dims
        for tensor_idx, tensor_strides in enumerate(self.strides[0]):
            if len(tensor_strides) != num_dims:
                tensor_names = ["in0", "in1", "out"] if is_binary else ["in", "out"]
                raise ValueError(
                    f"strides[0][{tensor_idx}] ({tensor_names[tensor_idx]}) length "
                    f"({len(tensor_strides)}) must match number of dimensions ({num_dims})."
                )

        # Unary-specific validations
        if is_unary:
            # All dim_types must be 'c' for unary operations
            for i, dt in enumerate(self.dim_types):
                if dt != DimType.c:
                    raise ValueError(
                        f"For unary operations, all dim_types must be etops.dim.c. "
                        f"dim_types[{i}] is {dt}."
                    )

            # prim_first and prim_last must be 'none' for unary
            if self.prim_first != PrimType.none:
                raise ValueError(
                    f"For unary operations, prim_first must be etops.prim.none, "
                    f"got {self.prim_first}."
                )
            if self.prim_last != PrimType.none:
                raise ValueError(
                    f"For unary operations, prim_last must be etops.prim.none, "
                    f"got {self.prim_last}."
                )

        # Binary-specific validations
        if is_binary:
            # Validate prim_first is compatible
            if self.prim_first not in [PrimType.none, PrimType.zero]:
                raise ValueError(
                    f"For binary contractions, prim_first must be etops.prim.none or "
                    f"etops.prim.zero, got {self.prim_first}."
                )

            # Validate prim_last is compatible
            if self.prim_last not in [PrimType.none, PrimType.relu]:
                raise ValueError(
                    f"For binary contractions, prim_last must be etops.prim.none or "
                    f"etops.prim.relu, got {self.prim_last}."
                )

    def to_json(self, indent: Optional[int] = None) -> str:
        """
        Serialize this configuration to a JSON string.

        Args:
            indent: Number of spaces for indentation. If None, output is compact.

        Returns:
            JSON string representation of the configuration.

        Example:
            >>> config = TensorOperationConfig(...)
            >>> json_str = config.to_json(indent=2)
            >>> print(json_str)
            {
              "backend": "tpp",
              "data_type": "float32",
              "prim_first": "zero",
              ...
            }
        """
        data = {
            "backend": self.backend,
            "data_type": self.data_type.name,
            "prim_first": self.prim_first.name,
            "prim_main": self.prim_main.name,
            "prim_last": self.prim_last.name,
            "dim_types": [dt.name for dt in self.dim_types],
            "exec_types": [et.name for et in self.exec_types],
            "dim_sizes": list(self.dim_sizes),
            "strides": [[list(tensor) for tensor in level] for level in self.strides],
        }
        return json.dumps(data, indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "TensorOperationConfig":
        """
        Deserialize a JSON string to a TensorOperationConfig.

        Args:
            json_str: JSON string representation of a configuration.

        Returns:
            A new TensorOperationConfig instance.

        Raises:
            ValueError: If the JSON is invalid or missing required fields.

        Example:
            >>> json_str = '{"backend": "tpp", "data_type": "float32", ...}'
            >>> config = TensorOperationConfig.from_json(json_str)
        """
        data = json.loads(json_str)

        # Required fields
        required = [
            "backend",
            "data_type",
            "prim_first",
            "prim_main",
            "prim_last",
            "dim_types",
            "exec_types",
            "dim_sizes",
            "strides",
        ]
        missing = [f for f in required if f not in data]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")

        # Helper to get enum value by name
        def get_enum(enum_type, name):
            try:
                return getattr(enum_type, name)
            except AttributeError:
                raise ValueError(f"Invalid {enum_type.__name__} value: '{name}'")

        return cls(
            backend=data["backend"],
            data_type=get_enum(DataType, data["data_type"]),
            prim_first=get_enum(PrimType, data["prim_first"]),
            prim_main=get_enum(PrimType, data["prim_main"]),
            prim_last=get_enum(PrimType, data["prim_last"]),
            dim_types=tuple(get_enum(DimType, dt) for dt in data["dim_types"]),
            exec_types=tuple(get_enum(ExecType, et) for et in data["exec_types"]),
            dim_sizes=tuple(data["dim_sizes"]),
            strides=tuple(
                tuple(tuple(tensor) for tensor in level) for level in data["strides"]
            ),
        )

    def save(self, path: str, indent: Optional[int] = 2) -> None:
        """
        Save this configuration to a JSON file.

        Args:
            path: File path to write to.
            indent: Number of spaces for indentation. Default is 2 for readability.

        Example:
            >>> config.save("my_config.json")
        """
        with open(path, "w") as f:
            f.write(self.to_json(indent=indent))

    @classmethod
    def load(cls, path: str) -> "TensorOperationConfig":
        """
        Load a configuration from a JSON file.

        Args:
            path: File path to read from.

        Returns:
            A new TensorOperationConfig instance.

        Example:
            >>> config = TensorOperationConfig.load("my_config.json")
        """
        with open(path, "r") as f:
            return cls.from_json(f.read())


__all__ = ["TensorOperationConfig"]
