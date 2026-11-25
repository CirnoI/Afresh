from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from iilslib.caster import caster_site_command, validate_name_map


class NameMapTest(unittest.TestCase):
    def test_caster_command_forwards_mapping(self):
        command = [str(value) for value in caster_site_command(
            Path('caster-site'), Path('loci.txt'), Path('guide.tre'), 4,
            mapping_path=Path('names.tsv'), seed=7,
        )]
        self.assertEqual(command[command.index('-a') + 1], 'names.tsv')

    def test_name_map_rejects_malformed_and_duplicate_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            malformed = root / 'malformed.tsv'
            malformed.write_text('copy_a species_a extra\n')
            with self.assertRaisesRegex(ValueError, 'exactly two fields'):
                validate_name_map(malformed)

            duplicate = root / 'duplicate.tsv'
            duplicate.write_text('copy_a species_a\ncopy_a species_a\n')
            with self.assertRaisesRegex(ValueError, 'duplicate sequence name'):
                validate_name_map(duplicate)


if __name__ == '__main__':
    unittest.main()
