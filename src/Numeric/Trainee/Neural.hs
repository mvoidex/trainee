{-# LANGUAGE GADTs, FlexibleContexts, RankNTypes #-}

module Numeric.Trainee.Neural (
	Net,
	(⭃),
	fc, conv, conv2,

	sigma, relu, softplus,
	summator, biaser, activator, convolver, convolver2,
	normVar,

	module Numeric.Trainee.Types,
	module Numeric.Trainee.Learnee
	) where

import Prelude hiding ((.), id)

import Control.Monad (replicateM, liftM2)
import Data.Random
import Numeric.LinearAlgebra hiding (conv, conv2)

import Numeric.Trainee.Types
import Numeric.Trainee.Gradee
import Numeric.Trainee.Learnee

type Net a = Learnee (Vector a) (Vector a)

(⭃) ∷ MonadRandom m ⇒ m (Learnee a b) → m (Learnee b c) → m (Learnee a c)
n ⭃ l = liftM2 (⇉) n l

-- | Fully connected layer
fc ∷ (MonadRandom m, Distribution Normal a, Numeric a, Num (Vector a), Parametric a) ⇒ Unary (Vector a) → Int → Int → m (Net a)
fc f inputs outputs = do
	s ← summator inputs outputs
	b ← biaser outputs
	return $ s ⇉ b ⇉ activator f

-- | Convolution layer 1-d
conv ∷ (MonadRandom m, Distribution Normal a, Numeric a, Num (Vector a), Parametric a) ⇒ Unary (Vector a) → Int → m (Net a)
conv f w = do
	c ← convolver w
	b ← do
		bs ← runRVar normVar StdRandom
		return $ learnee biasVec bs
	return $ c ⇉ b ⇉ activator f

-- | Convolution layer 2-d
conv2 ∷ (MonadRandom m, Distribution Normal a, Numeric a, Num (Vector a), Parametric a) ⇒ Unary (Matrix a) → (Int, Int) → m (Learnee (Matrix a) (Matrix a))
conv2 f (w, h) = do
	c ← convolver2 w h
	b ← do
		bs ← runRVar normVar StdRandom
		return $ learnee biasMat bs
	return $ c ⇉ b ⇉ activator f

sigma ∷ Floating a ⇒ a → a
sigma t = 1 / (1 + exp (negate t))

relu ∷ (Fractional a) ⇒ a → a
relu t = 0.5 * (1 + signum t) * t

softplus ∷ Floating a ⇒ a → a
softplus t = log (1 + exp t)

summator ∷ (MonadRandom m, Distribution Normal a, Fractional a, Numeric a, Num (Vector a), Parametric a) ⇒ Int → Int → m (Net a)
summator inputs outputs = do
	ws ← runRVar (replicateM (inputs * outputs) normVar) StdRandom
	return $ learnee matVec ((outputs >< inputs) ws)

biaser ∷ (MonadRandom m, Distribution Normal a, Numeric a, Num (Vector a), Parametric a) ⇒ Int → m (Net a)
biaser sz = do
	bs ← runRVar (replicateM sz normVar) StdRandom
	return $ learnee odot (fromList bs)

activator ∷ Num a ⇒ Unary a → Learnee a a
activator f = computee (unary f)

convolver ∷ (MonadRandom m, Distribution Normal a, Numeric a, Num (Vector a), Parametric a) ⇒ Int → m (Net a)
convolver w = do
	ws ← runRVar (replicateM w normVar) StdRandom
	return $ learnee corrVec (fromList ws)

convolver2 ∷ (MonadRandom m, Distribution Normal a, Fractional a, Numeric a, Num (Vector a), Parametric a) ⇒ Int → Int → m (Learnee (Matrix a) (Matrix a))
convolver2 w h = do
	ws ← runRVar (replicateM (w * h) normVar) StdRandom
	return $ learnee corrMat ((w >< h) ws)

normVar ∷ (Distribution Normal a, Fractional a) ⇒ RVar a
normVar = normal 0.0 0.25
