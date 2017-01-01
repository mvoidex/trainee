{-# LANGUAGE FlexibleContexts #-}

module Numeric.Trainee.Data.Csv (
	parseCsv, parseCsvFile
	) where

import Control.Exception (throwIO)
import Data.ByteString.Lazy (ByteString)
import qualified Data.ByteString.Lazy.Char8 as L
import Data.Foldable (toList)
import Data.Csv
import Numeric.LinearAlgebra hiding (toList)

import Numeric.Trainee.Types
import Numeric.Trainee.Data.Common

parseCsv ∷ Container Vector a ⇒ ByteString → Sample [Attr String a] [Attr String a] → Either String [Sample (Vector a) (Vector a)]
parseCsv dat (Sample attrsIn attrsOut) = do
	entries ← decode NoHeader dat
	let
		toSample entry = (⇢) <$> appAttrs attrsIn icols <*> appAttrs attrsOut ocols where
			(icols, ocols) = splitAt (length attrsIn) entry
	mapM toSample $ toList entries

parseCsvFile ∷ Container Vector a ⇒ FilePath → Sample [Attr String a] [Attr String a] → IO [Sample (Vector a) (Vector a)]
parseCsvFile fpath attrs = do
	cts ← L.readFile fpath
	case parseCsv cts attrs of
		Left e → throwIO $ userError $ "Error parsing " ++ fpath ++ ": " ++ e
		Right rs → return rs