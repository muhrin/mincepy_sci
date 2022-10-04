# -*- coding: utf-8 -*-
"""Module that provides interoperability between ase and mincepy"""

import uuid

import ase.cell
import ase.db.row
import ase.calculators.calculator as ase_calculator
import mincepy

__all__ = "AtomsHelper", "CellHelper"


class AtomsHelper(mincepy.TypeHelper):
    TYPE = ase.Atoms
    TYPE_ID = uuid.UUID("ad4ca7ae-6ebc-4594-947d-ac42f5d96c1f")
    INJECT_CREATION_TRACKING = True

    def __init__(self, load_original_calculator=False):
        super().__init__()
        self._load_original_calculator = load_original_calculator

    def yield_hashables(
        self, atoms: ase.Atoms, hasher
    ):  # pylint: disable=arguments-differ
        yield from hasher.yield_hashables(atoms.cell)
        yield from hasher.yield_hashables(atoms.pbc)
        yield from hasher.yield_hashables(atoms.positions)
        yield from hasher.yield_hashables(atoms.numbers)

    def eq(self, one, other) -> bool:
        if not (isinstance(one, ase.Atoms) and isinstance(other, ase.Atoms)):
            return False

        return (
            len(one) == len(other)
            and (one.positions == other.positions).all()
            and (one.numbers == other.numbers).all()
            and (one.cell == other.cell).all()
            and (one.pbc == other.pbc).all()
        )

    def save_instance_state(
        self, atoms: ase.Atoms, saver
    ):  # pylint: disable=arguments-differ
        return ase.db.row.atoms2dict(atoms)

    def load_instance_state(
        self, atoms: ase.Atoms, saved_state, _referencer
    ):  # pylint: disable=arguments-differ
        """Much of this is taken from ase.db.row.AtomsRow.toatoms() and therefore may
        need to be updated if the ase code changes"""
        row = ase.db.row.AtomsRow(saved_state)
        atoms.__init__(
            row.numbers,  # pylint: disable=no-member
            row.positions,  # pylint: disable=no-member
            cell=row.cell,
            pbc=row.pbc,
            magmoms=row.get("initial_magmoms"),
            charges=row.get("initial_charges"),
            tags=row.get("tags"),
            masses=row.get("masses"),
            momenta=row.get("momenta"),
            constraint=row.constraints,
        )

        if self._load_original_calculator:
            try:
                atoms.calc = ase_calculator.get_calculator_class(row.calculator)(
                    **row.get("calculator_parameters", {})
                )
            except AttributeError:
                pass  # No calculator
        else:
            results = {}
            for prop in ase_calculator.all_properties:
                if prop in row:
                    results[prop] = row[prop]
            if results:
                atoms.calc = ase.calculators.singlepoint.SinglePointCalculator(
                    atoms, **results
                )
                atoms.calc.name = row.get("calculator", "unknown")

        atoms.info = {}
        atoms.info["unique_id"] = row.unique_id  # pylint: disable=no-member
        if row.key_value_pairs:
            atoms.info["key_value_pairs"] = row.key_value_pairs
        data = row.get("data")
        if data:
            atoms.info["data"] = data

        return atoms


class CellHelper(mincepy.TypeHelper):
    TYPE = ase.cell.Cell
    TYPE_ID = uuid.UUID("4eea34e2-df87-420e-b51e-7d015bb1d3cb")
    INJECT_CREATION_TRACKING = True

    def yield_hashables(
        self, cell: ase.cell.Cell, hasher
    ):  # pylint: disable=arguments-differ
        yield from hasher.yield_hashables(cell.array.tolist())

    def eq(self, one, other) -> bool:
        if not isinstance(other, ase.cell.Cell):
            return False

        return (one == other).all()

    def save_instance_state(
        self, cell: ase.cell.Cell, _referencer
    ):  # pylint: disable=arguments-differ
        # Here we emulate what cell's todict method started to do in 3.20 because
        # otherwise we get a deprecation warning on account of pbc being removed
        # In case todict changes we should update this correspondingly (eventually
        # once 3.19 is no longer used this could go back to just calling todict())
        return dict(array=cell.array)

    def load_instance_state(
        self, cell: ase.cell.Cell, saved_state, _referencer
    ):  # pylint: disable=arguments-differ
        cell.__init__(saved_state["array"])
        return cell


TYPES = CellHelper(), AtomsHelper()
