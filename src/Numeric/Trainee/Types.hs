{-# LANGUAGE GADTs, RankNTypes, MultiParamTypeClasses, FlexibleInstances, UndecidableInstances, KindSignatures, ScopedTypeVariables, ConstraintKinds, TemplateHaskell, GeneralizedNewtypeDeriving, TypeFamilies #-}

module Numeric.Trainee.Types (
	Gradee(..),
	Ow(..), NoParams(..),
	params, forwardPass, Learnee(..),
	Cost, HasNorm(..),
	Sample(..), (⇢),
	Samples, samples,

	module Numeric.Trainee.Params
	) where

import Prelude hiding (id, (.))

import Control.Category
import Control.DeepSeq
import Control.Lens
import Data.Typeable (cast)
import qualified Data.Vector as V
import Numeric.LinearAlgebra (Normed(norm_1), R, Vector)

import Numeric.Trainee.Params

newtype Gradee a b = Gradee {
	runGradee ∷ Lens' a b }

instance Category Gradee where
	id = Gradee id
	Gradee f . Gradee g = Gradee (g . f)

class Category a ⇒ Ow a where
	first ∷ a b c → a (b, d) (c, d)
	first = flip stars id

	second ∷ a b c → a (d, b) (d, c)
	second = stars id

	stars ∷ a b c → a b' c' → a (b, b') (c, c')
	vstars ∷ a b c → a (V.Vector b) (V.Vector c)

instance Ow Gradee where
	stars (Gradee f) (Gradee g) = Gradee $ lens g' s' where
		g' (x, y) = (view f x, view g y)
		s' (x, y) (x', y') = (set f x' x, set g y' y)

	vstars (Gradee f) = Gradee $ lens g' s' where
		g' = V.map (view f)
		s' = V.zipWith (flip (set f))

data NoParams = NoParams deriving (Eq, Ord, Read, Enum, Bounded)

instance NFData NoParams where
	rnf NoParams = ()

instance Show NoParams where
	show NoParams = ""

instance Num NoParams where
	_ + _ = NoParams
	_ * _ = NoParams
	abs _ = NoParams
	signum _ = NoParams
	fromInteger _ = NoParams
	negate _ = NoParams

instance Fractional NoParams where
	fromRational _ = NoParams
	recip _ = NoParams

data Learnee a b = Learnee {
	_params ∷ Params,
	_forwardPass ∷ Params → a → (b, b → (Params, a)) }

makeLenses ''Learnee

instance NFData (Learnee a b) where
	rnf (Learnee ws _) = rnf ws

instance Show (Learnee a b) where
	show (Learnee ws _) = show ws

instance Category Learnee where
	id = Learnee (Params NoParams) fn where
		fn _ x = (x, const (Params NoParams, x))
	Learnee rws g . Learnee lws f = lws `deepseq` rws `deepseq` Learnee (Params (lws, rws)) h where
		h ws x = case cast ws of
			(Just (lws', rws')) → x `seq` y `seq` lws' `deepseq` rws' `deepseq` (z, up) where
				(y, f') = f lws' x
				(z, g') = g rws' y
				up dz = dz `seq` dy `seq` lws'' `deepseq` rws'' `deepseq` (Params (lws'', rws''), dx) where
					(rws'', dy) = g' dz
					(lws'', dx) = f' dy
			_ → error "learnee: (.): impossible"

-- instance Category Learnee where
-- 	id = Learnee NoParams fn where
-- 		fn _ x = (x, const (NoParams, x))
-- 	Learnee rws g . Learnee lws f = lws `deepseq` rws `deepseq` Learnee (PairParams lws rws) h where
-- 		h (PairParams lws' rws') x = x `seq` y `seq` lws' `deepseq` rws' `deepseq` (z, up) where
-- 			(y, f') = f lws' x
-- 			(z, g') = g rws' y
-- 			up dz = dz `seq` dy `seq` lws'' `deepseq` rws'' `deepseq` (PairParams lws'' rws'', dx) where
-- 				(rws'', dy) = g' dz
-- 				(lws'', dx) = f' dy

type Cost b = b → b → (b, b)

class HasNorm a where
	type Norm a
	norm ∷ a → Norm a

instance HasNorm Float where
	type Norm Float = Float
	norm = id

instance HasNorm Double where
	type Norm Double = Double
	norm = id

instance Normed (Vector a) ⇒ HasNorm (Vector a) where
	type Norm (Vector a) = R
	norm = norm_1

data Sample a b = Sample {
	sampleInput ∷ a,
	sampleOutput ∷ b }
		deriving (Eq, Ord)

(⇢) ∷ a → b → Sample a b
(⇢) = Sample

instance Bifunctor Sample where
	bimap f g (Sample x y) = Sample (f x) (g y)

instance (Show a, Show b) ⇒ Show (Sample a b) where
	show (Sample xs ys) = show xs ++ " => " ++ show ys

instance (NFData a, NFData b) ⇒ NFData (Sample a b) where
	rnf (Sample i o) = rnf i `seq` rnf o

type Samples a b = V.Vector (Sample a b)

samples ∷ [Sample a b] → Samples a b
samples = V.fromList
