{-# LANGUAGE FlexibleInstances, FlexibleContexts #-}

module DataSet (
	Example, (⇢),
	Attr,
	val, scaleBy, className, enumValue, bool,
	toSample,
	parseData, parseFile
	) where

import Prelude.Unicode

import Control.Arrow ((&&&))
import Control.Exception (throwIO)
import qualified Data.ByteString.Lazy.Char8 as L
import Data.List (findIndex, intercalate)
import Data.Csv
import Data.Foldable (toList)
import Numeric.LinearAlgebra hiding (toList)

import Numeric.Trainee.Types

type Example a = Sample [a] a

(⇢) ∷ a → b → Sample a b
(⇢) = (,)

type Attr a = String → Vector a

val ∷ (Read a, Container Vector a) ⇒ Attr a
val = fromList ∘ return ∘ read

scaleBy ∷ Linear a Vector ⇒ a → Attr a → Attr a
scaleBy v fn = scale v ∘ fn

className ∷ (Num a, Container Vector a) ⇒ [String] → Attr a
className cls name = case findIndex (≡ name) cls of
	Nothing → error $ "Invalid class value '" ++ name ++ "', expected " ++ intercalate ", " ["'" ++ c ++ "'" | c ← cls]
	Just idx → assoc (length cls) 0 [(idx, 1)]

enumValue ∷ (Fractional a, Container Vector a) ⇒ [String] → Attr a
enumValue vals name = case findIndex (≡ name) vals of
	Nothing → error $ "Invalid enum value '" ++ name ++ "', expected " ++ intercalate ", " ["'" ++ v ++ "'" | v ← vals]
	Just idx → fromList [fromIntegral idx / fromIntegral (length vals - 1)]

bool ∷ (Fractional a, Container Vector a) ⇒ Attr a
bool = enumValue ["f", "t"]

toSample ∷ Container Vector a ⇒ Example (Attr a) → Example String → Sample (Vector a) (Vector a)
toSample (xs', ys') (xs, ys) = (apply xs' xs, apply (return ys') (return ys)) where
	apply ∷ Container Vector a ⇒ [String → Vector a] → [String] → Vector a
	apply attrs' vals' = vjoin $ zipWith ($) attrs' vals'

parseData ∷ Container Vector a ⇒ L.ByteString → Example (Attr a) → Either String [Sample (Vector a) (Vector a)]
parseData dat attrs = do
	entries ← decode NoHeader dat
	return $ map (toSample attrs ∘ (init &&& last)) $ toList entries

parseFile ∷ Container Vector a ⇒ FilePath → Example (Attr a) → IO [Sample (Vector a) (Vector a)]
parseFile fpath attrs = do
	cts ← L.readFile fpath
	case parseData cts attrs of
		Left e → throwIO $ userError $ "Error parsing '" ++ fpath ++ "': " ++ e
		Right rs → return rs
