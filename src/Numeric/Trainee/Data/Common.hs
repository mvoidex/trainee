{-# LANGUAGE FlexibleContexts #-}

module Numeric.Trainee.Data.Common (
	Attr, appAttrs, val, read_, enum_, class_
	) where

import Prelude.Unicode

import Control.Monad
import Data.List
import Numeric.LinearAlgebra
import Text.Read (readMaybe)

type Attr a b = a → Either String (Vector b)

appAttrs ∷ Container Vector b ⇒ [Attr a b] → [a] → Either String (Vector b)
appAttrs as xs = vjoin <$> zipWithM ($) as xs

val ∷ Container Vector a ⇒ Attr a a
val = return ∘ fromList ∘ return
read_ ∷ (Read a, Container Vector a) ⇒ Attr String a

read_ s = case readMaybe s of
	Nothing → Left $ "error parsing value: " ++ s
	Just v → val v

enum_ ∷ (Fractional a, Container Vector a) ⇒ [String] → Attr String a
enum_ names name = case findIndex (≡ name) names of
	Nothing → Left $ "invalid enum value: " ++ name ++ ", expected " ++ intercalate ", " names
	Just idx → val $ fromIntegral idx / fromIntegral (length names - 1)

class_ ∷ (Num a, Container Vector a) ⇒ [String] → Attr String a
class_ names name = case findIndex (≡ name) names of
	Nothing → Left $ "invalid class value: " ++ name ++ ", expected " ++ intercalate ", " names
	Just idx → return $ assoc (length names) 0 [(idx, 1)]
