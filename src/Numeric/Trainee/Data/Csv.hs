{-# LANGUAGE FlexibleContexts #-}

module Numeric.Trainee.Data.Csv (
	parseCsv, parseCsvFile
	) where

import Prelude.Unicode

import Control.DeepSeq
import Control.Exception (throwIO)
import qualified Data.Vector as V
import qualified Data.Text as T
import qualified Data.Text.IO as T
import Numeric.LinearAlgebra hiding (toList)

import Numeric.Trainee.Types
import Numeric.Trainee.Data.Common

parseCsv ∷ Bool → T.Text → CsvM (Sample (Vector a) (Vector a)) → Either String (Samples (Vector a) (Vector a))
parseCsv header' dat p = mapM (runCsvM p ∘ V.fromList ∘ T.split (≡ ',')) $ V.fromList $ dropHeader $ T.lines dat where
	dropHeader
		| header' = tail
		| otherwise = id

parseCsvFile ∷ Bool → FilePath → CsvM (Sample (Vector a) (Vector a)) → IO (Samples (Vector a) (Vector a))
parseCsvFile header' fpath p = do
	cts ← T.readFile fpath
	case parseCsv header' cts p of
		Left e → throwIO $ userError $ "Error parsing " ++ fpath ++ ": " ++ e
		Right rs → return $!! rs
